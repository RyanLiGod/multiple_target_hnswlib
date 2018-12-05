#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include "hnswlib.h"
#include "visited_list_pool.h"

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
public:
    HierarchicalNSW(SpaceInterface<dist_t>* s) {
    }

    HierarchicalNSW(SpaceInterface<dist_t>* s, const std::string& location, bool nmslib = false, size_t max_elements = 0) {
        loadIndex(location, s, max_elements);
    }

    HierarchicalNSW(SpaceInterface<dist_t>* s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100)
        : link_list_locks_(max_elements), element_levels_(max_elements) {
        max_elements_ = max_elements;

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();  // 数据的维数 dim
        // 在构造期间为每个新元素创建的双向链接的数量
        M_ = M;
        maxM_ = M_;
        // 在第 0 层链接数量翻倍
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);  // 使用种子 random_seed 充值 level_generator_ 的状态

        // 第 0 层链接的存储大小 sizeof(unsigned int) == 4
        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        // 第 0 层每个元素存储空间：链接大小 + 数据大小 + label 大小
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        // offsetData_ 是在元素存储空间内，数据存储地址的偏移量
        offsetData_ = size_links_level0_;
        // label_offset_ 是在元素存储空间内，label存储地址的偏移量
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        // 在第 0 层需要的存储空间
        data_level0_memory_ = (char*)malloc(max_elements_ * size_data_per_element_);

        // nullptr -> 空指针
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        // 创建 1 个容量为 max_elements 的访问列表池
        visited_list_pool_ = new VisitedListPool(1, max_elements);

        // 初始化首节点特殊处理
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        /**
             * linkLists_是链接列表的首地址， sizeof(void*)的含义就是获取一个指针的大小，(char **)指向二维字符指针
             * 记录每个元素在每一层之中的链接，所以是一个二维数组
             * linkLists_[internal_id] = [level1的链接地址，level2的链接地址，...，levelN的链接地址]
             */
        linkLists_ = (char**)malloc(sizeof(void*) * max_elements_);
        // 每个元素的链接大小
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        // 假设 M 大于 16 值域约 [0.2, 0.36]
        mult_ = 1 / log(1.0 * M_);
        // 假设 M 大于 16 值域约 [2.7, 4.6]
        revSize_ = 1.0 / mult_;
    }

    struct CompareByFirst {
        // constexpr 常量表达式
        constexpr bool operator()(std::pair<dist_t, tableint> const& a, std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    ~HierarchicalNSW() {
        free(data_level0_memory_);
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        delete visited_list_pool_;
    }

    size_t max_elements_;            // 设定的最大元素数量
    size_t cur_element_count;        // 目前元素数量
    size_t size_data_per_element_;   // 每条元素的大小（链接 + 数据 + label）
    size_t size_links_per_element_;  // 每条元素的链接大小

    size_t M_;
    size_t maxM_;
    size_t maxM0_;
    size_t ef_construction_;

    double mult_, revSize_;
    int maxlevel_;

    VisitedListPool* visited_list_pool_;
    std::mutex cur_element_count_guard_;

    std::vector<std::mutex> link_list_locks_;
    tableint enterpoint_node_;

    size_t size_links_level0_;
    size_t offsetData_, offsetLevel0_;

    char* data_level0_memory_;
    char** linkLists_;
    std::vector<int> element_levels_;

    size_t data_size_;
    size_t label_offset_;
    DISTFUNC<dist_t> fstdistfunc_;
    void* dist_func_param_;  // 数据的维数 dim

    std::default_random_engine level_generator_;

    // 获取外部 label 的值
    inline labeltype getExternalLabel(tableint internal_id) const {
        return *((labeltype*)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_));
    }

    // 获取外部 label 的地址
    inline labeltype* getExternalLabeLp(tableint internal_id) const {
        return (labeltype*)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }

    // 通过内部 id 获取 data 地址
    inline char* getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }

    /**
     * 随机数种子 level_generator_ = 100, reverse_size = 1 / log(1.0 * M_) 假设 M 大于 16 值域 [0.2, 0.36]
     * M 值一般取 5-100，log()的底是 e
     */
    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
    }

    /**
     * enterpoint_id 开始的点
     * data_point 需要插入的点的向量
     * layer 层数
     */
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint enterpoint_id, void* data_point, int layer) {
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(enterpoint_id), dist_func_param_);

        top_candidates.emplace(dist, enterpoint_id);
        candidateSet.emplace(-dist, enterpoint_id);
        visited_array[enterpoint_id] = visited_array_tag;
        dist_t lowerBound = dist;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();

            if ((-curr_el_pair.first) > lowerBound) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

            int* data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0)
                data = (int*)(data_level0_memory_ + curNodeNum * size_data_per_element_ + offsetLevel0_);
            else
                data = (int*)(linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            int size = *data;
            tableint* datal = (tableint*)(data + 1);
            _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);

            for (int j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
                _mm_prefetch((char*)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
                visited_array[candidate_id] = visited_array_tag;
                char* currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.top().first > dist1 || top_candidates.size() < ef_construction_) {
                    candidateSet.emplace(-dist1, candidate_id);
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
                    top_candidates.emplace(dist1, candidate_id);
                    if (top_candidates.size() > ef_construction_) {
                        top_candidates.pop();
                    }
                    lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(tableint ep_id, const void* data_point, size_t ef) const {
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);

        top_candidates.emplace(dist, ep_id);
        candidate_set.emplace(-dist, ep_id);
        visited_array[ep_id] = visited_array_tag;
        dist_t lower_bound = dist;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

            if ((-current_node_pair.first) > lower_bound) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int* data = (int*)(data_level0_memory_ + current_node_id * size_data_per_element_ + offsetLevel0_);
            int size = *data;
            _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char*)(data + 2), _MM_HINT_T0);

            for (int j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                _mm_prefetch((char*)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char* currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    if (top_candidates.top().first > dist || top_candidates.size() < ef) {
                        candidate_set.emplace(-dist, candidate_id);
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ + offsetLevel0_, _MM_HINT_T0);

                        top_candidates.emplace(dist, candidate_id);

                        if (top_candidates.size() > ef) {
                            top_candidates.pop();
                        }
                        lower_bound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    // 探索性地找到小于M个邻居
    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>& top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }
        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;
            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                    fstdistfunc_(getDataByInternalId(second_pair.second),
                                 getDataByInternalId(curent_pair.second),
                                 dist_func_param_);
                ;
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    // 获取第 0 层的 internal_id 的链接列表地址
    linklistsizeint* get_linklist0(tableint internal_id) {
        return (linklistsizeint*)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    };

    linklistsizeint* get_linklist0(tableint internal_id, char* data_level0_memory_) {
        return (linklistsizeint*)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    };

    // 获取第 level 层 internal_id 的链接列表地址
    linklistsizeint* get_linklist(tableint internal_id, int level) {
        return (linklistsizeint*)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    };

    /**
     * mutuallyConnectNewElement() 连接元素
     * data_point 当前需要插入的点
     * cur_c 当前的元素个数
     * top_candidates 需要连接的点
     * level 层数
     */
    void mutuallyConnectNewElement(void* data_point, tableint cur_c, std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates, int level) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        // reserve():重新申请并改变当前vector对象的总空间（_capacity）大小'
        // 如果在事先预见到有较大空间需求，就可以先用reserve预留一定的空间，避免内存重复分配和大量的数据搬移，提高了效率。
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }
        {
            linklistsizeint* ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            *ll_cur = selectedNeighbors.size();
            tableint* data = (tableint*)(ll_cur + 1);

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx])
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint* ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);
            size_t sz_link_list_other = *ll_other;

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint* data = (tableint*)(ll_other + 1);
            if (sz_link_list_other < Mcurmax) {
                data[sz_link_list_other] = cur_c;
                *ll_other = sz_link_list_other + 1;
            } else {
                // finding the "weakest" element to replace it with the new one
                dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);
                // Heuristic:
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                candidates.emplace(d_max, cur_c);

                for (size_t j = 0; j < sz_link_list_other; j++) {
                    candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]), dist_func_param_), data[j]);
                }

                getNeighborsByHeuristic2(candidates, Mcurmax);

                int indx = 0;
                while (candidates.size() > 0) {
                    data[indx] = candidates.top().second;
                    candidates.pop();
                    indx++;
                }
                *ll_other = indx;
                // Nearest K:
                /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
            }
        }
    }

    std::mutex global;
    size_t ef_;

    void setEf(size_t ef) {
        ef_ = ef;
    }

    std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void* query_data, int k) {
        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (size_t level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                int* data;
                data = (int*)(linkLists_[currObj] + (level - 1) * size_links_per_element_);
                int size = *data;
                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        //std::priority_queue< std::pair< dist_t, tableint  >> top_candidates = searchBaseLayer(currObj, query_data, 0);
        std::priority_queue<std::pair<dist_t, tableint>> top_candidates = searchBaseLayerST(currObj, query_data, ef_);
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        return top_candidates;
    };

    void saveIndex(const std::string& location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }

    void loadIndex(const std::string& location, SpaceInterface<dist_t>* s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        /// Legacy, check that everything is ok

        bool old_index = false;

        auto pos = input.tellg();
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                old_index = true;
                break;
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // check if file is ok, if not this is either corrupted or old index
        if (input.tellg() != total_filesize)
            old_index = true;

        if (old_index) {
            std::cerr << "Warning: loading of old indexes will be deprecated before 2019.\n"
                      << "Please resave the index in the new format.\n";
        }
        input.clear();
        input.seekg(pos, input.beg);

        data_level0_memory_ = (char*)malloc(max_elements * size_data_per_element_);
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        if (old_index)
            input.seekg(((max_elements_ - cur_element_count) * size_data_per_element_), input.cur);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);

        visited_list_pool_ = new VisitedListPool(1, max_elements);

        linkLists_ = (char**)malloc(sizeof(void*) * max_elements);
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;

                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char*)malloc(linkListSize);
                input.read(linkLists_[i], linkListSize);
            }
        }
        input.close();

        return;
    }

    void addPoint(void* data_point, labeltype label) {
        addPoint(data_point, label, -1);
    };

    tableint addPoint(void* data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            std::unique_lock<std::mutex> lock(cur_element_count_guard_);
            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            };
            cur_c = cur_element_count;
            cur_element_count++;
        }
        std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);  // 获取一个层数
        std::cout << "curlevel: " << curlevel << std::endl;

        if (level > 0)
            curlevel = level;

        // 在 element_levels_ 设定了该元素的层数
        element_levels_[cur_c] = curlevel;

        std::unique_lock<std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;  // 首节点maxlevel_==-1
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;  // 首节点enterpoint_node_==-1，之后等于cur_c-1

        // 清空当前要加入的点需要占据的内存以便数据插入
        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        // 如果 curlevel 层数不为 0，就初始化 linkLists_[cur_c]
        if (curlevel) {
            linkLists_[cur_c] = (char*)malloc(size_links_per_element_ * curlevel + 1);
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                // 获取data_point和上一个插入的点的距离
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        int* data;
                        std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                        /**
                         * data 当前层的链接地址
                         * linkLists_[currObj] currObj的链接地址
                         */
                        data = (int*)(linkLists_[currObj] + (level - 1) * size_links_per_element_);
                        int size = *data;
                        tableint* datal = (tableint*)(data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            // 插入点和候选点的距离
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            // 如果d小于curdist，就更改curdist
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)
                    throw std::runtime_error("Level error");

                /**
                 * enterpoint_id 开始的点: currObj
                 * data_point 需要插入的点的向量: data_point
                 * layer 层数: level
                 */
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(currObj, data_point, level);
                mutuallyConnectNewElement(data_point, cur_c, top_candidates, level);
            }

        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;  // 返回当前数据个数
    };

    std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(const void* query_data, size_t k) const {
        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                int* data;
                data = (int*)(linkLists_[currObj] + (level - 1) * size_links_per_element_);
                int size = *data;
                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayerST(currObj, query_data, std::max(ef_, k));
        std::priority_queue<std::pair<dist_t, labeltype>> results;
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            results.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return results;
    };
};

}  // namespace hnswlib
