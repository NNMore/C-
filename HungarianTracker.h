#ifndef HUNGARIAN_TRACKER_H
#define HUNGARIAN_TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>

class HungarianTracker {
public:
    HungarianTracker(double distance_threshold = 50.0);
    std::vector<std::vector<int>> update(const std::vector<cv::Rect>& objects_rect, const cv::Mat& image);
    std::unordered_map<int, cv::Mat> retain_histograms();

private:
    cv::Mat compute_histogram(const cv::Mat& image, const cv::Rect& rect);

    std::unordered_map<int, cv::Point> center_points; // Хранит центр объектов
    std::unordered_map<int, cv::Mat> histograms; // Хранит гистограммы объектов
    int id_count; // Уникальный идентификатор
    double distance_threshold; // Пороговое значение для определения "достаточной близости"
};

#endif // HUNGARIAN_TRACKER_H