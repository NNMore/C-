#include "HungarianTracker.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

HungarianTracker::HungarianTracker(double distance_threshold)
    : id_count(0), distance_threshold(distance_threshold) {}

cv::Mat HungarianTracker::compute_histogram(const cv::Mat& image, const cv::Rect& rect) {
    cv::Mat roi = image(rect);
    cv::cvtColor(roi, roi, cv::COLOR_BGR2HSV);
    cv::Mat hist;

    int h_bins = 8, s_bins = 8;
    int histSize[] = { h_bins, s_bins };
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };

    cv::calcHist(&roi, 1, channels, cv::Mat(), hist, 2, histSize, ranges);
    cv::normalize(hist, hist);
    return hist;
}

std::vector<std::vector<int>> HungarianTracker::update(const std::vector<cv::Rect>& objects_rect, const cv::Mat& image) {
    size_t num_objects = objects_rect.size();
    cv::Mat cost_matrix(num_objects, histograms.size(), CV_64F, cv::Scalar(0));

    // Заполнение матрицы затрат
    for (int i = 0; i < num_objects; ++i) {
        cv::Mat hist1 = compute_histogram(image, objects_rect[i]);
        int j = 0;
        for (const auto& [id, hist2] : histograms) {
            double cost = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);
            cost_matrix.at<double>(i, j) = cost;
            j++;
        }
    }

    // Поиск совпадений
    std::vector<int> row_ind(num_objects), col_ind(histograms.size());
    cv::reduce(cost_matrix, row_ind, 0, cv::REDUCE_MIN, CV_64F);
    cv::reduce(cost_matrix, col_ind, 1, cv::REDUCE_MIN, CV_64F);

    std::vector<std::vector<int>> objects_bbs_ids;

    for (size_t i = 0; i < num_objects; ++i) {
        bool found = false;
        for (size_t j = 0; j < histograms.size(); ++j) {
            if (cost_matrix.at<double>(i, j) < distance_threshold) {
                int id = std::distance(histograms.begin(), histograms.find(j));
                cv::Rect rect = objects_rect[i];
                cv::Point center((rect.x + rect.width / 2), (rect.y + rect.height / 2));
                center_points[id] = center;
                objects_bbs_ids.push_back({ rect.x, rect.y, rect.width, rect.height, id });
                found = true;
                break;
            }
        }

        if (!found) {
            cv::Rect rect = objects_rect[i];
            int new_id = id_count++;
            center_points[new_id] = cv::Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
            histograms[new_id] = compute_histogram(image, rect);
            objects_bbs_ids.push_back({ rect.x, rect.y, rect.width, rect.height, new_id });
        }
    }

    return objects_bbs_ids;
}

std::unordered_map<int, cv::Mat> HungarianTracker::retain_histograms() {
    return histograms;
}