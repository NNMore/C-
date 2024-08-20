#include "HungarianTracker.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

// Конструктор класса HungarianTracker
HungarianTracker::HungarianTracker(double distance_threshold)
    : id_count(0), distance_threshold(distance_threshold) {
    // Инициализация счетчика идентификаторов (id_count) и порога расстояния (distance_threshold)
}

// Метод для вычисления гистограммы в заданной области изображения
cv::Mat HungarianTracker::compute_histogram(const cv::Mat& image, const cv::Rect& rect) {
    // Получение региона интереса (ROI) из изображения
    cv::Mat roi = image(rect);
    // Преобразование цветового пространства из BGR в HSV
    cv::cvtColor(roi, roi, cv::COLOR_BGR2HSV);
    cv::Mat hist; // Переменная для хранения гистограммы

    // Определение параметров гистограммы
    int h_bins = 8, s_bins = 8; // Количество диапазонов для каждого канала
    int histSize[] = { h_bins, s_bins }; // Размеры гистограммы
    float h_ranges[] = { 0, 180 }; // Диапазон для канала H
    float s_ranges[] = { 0, 256 }; // Диапазон для канала S
    const float* ranges[] = { h_ranges, s_ranges }; // Массив диапазонов
    int channels[] = { 0, 1 }; // Каналы для построения гистограммы

    // Вычисление гистограммы для ROI
    cv::calcHist(&roi, 1, channels, cv::Mat(), hist, 2, histSize, ranges);
    // Нормализация гистограммы
    cv::normalize(hist, hist);
    return hist; // Возврат вычисленной гистограммы
}

// Метод для обновления положения объектов и сопоставления их с существующими
std::vector<std::vector<int>> HungarianTracker::update(const std::vector<cv::Rect>& objects_rect, const cv::Mat& image) {
    size_t num_objects = objects_rect.size(); // Получение количества объектов
    // Создание матрицы затрат для Венгерского алгоритма
    cv::Mat cost_matrix(num_objects, histograms.size(), CV_64F, cv::Scalar(0));

    // Заполнение матрицы затрат
    for (int i = 0; i < num_objects; ++i) {
        cv::Mat hist1 = compute_histogram(image, objects_rect[i]); // Вычисление гистограммы для текущего объекта
        int j = 0;
        // Проход по существующим гистограммам для вычисления стоимости
        for (const auto& [id, hist2] : histograms) {
            // Сравнение гистограмм и заполнение матрицы затрат
            double cost = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);
            cost_matrix.at<double>(i, j) = cost; // Запись стоимости в матрицу
            j++;
        }
    }

    // Поиск совпадений с использованием метода Венгерского алгоритма
    std::vector<int> row_ind(num_objects), col_ind(histograms.size());
    cv::reduce(cost_matrix, row_ind, 0, cv::REDUCE_MIN, CV_64F); // Минимизация по строкам
    cv::reduce(cost_matrix, col_ind, 1, cv::REDUCE_MIN, CV_64F); // Минимизация по столбцам

    std::vector<std::vector<int>> objects_bbs_ids; // Вектор для хранения всех объектов с их ID

    // Сопоставление объектов
    for (size_t i = 0; i < num_objects; ++i) {
        bool found = false; // Переменная для отслеживания, был ли найден матч
        for (size_t j = 0; j < histograms.size(); ++j) {
            // Проверка, если стоимость меньше порога
            if (cost_matrix.at<double>(i, j) < distance_threshold) {
                // Получаем ID объекта и обновляем координаты его центра
                int id = std::distance(histograms.begin(), histograms.find(j));
                cv::Rect rect = objects_rect[i];
                cv::Point center((rect.x + rect.width / 2), (rect.y + rect.height / 2));
                center_points[id] = center; // Сохраняем центр объекта
                // Добавляем объект с его ID в результирующий вектор
                objects_bbs_ids.push_back({ rect.x, rect.y, rect.width, rect.height, id });
                found = true; // Объект найден
                break; // Прерывание внутреннего цикла
            }
        }

        // Если объект не найден, создаем новый ID
        if (!found) {
            cv::Rect rect = objects_rect[i];
            int new_id = id_count++; // Генерация нового ID
            center_points[new_id] = cv::Point(rect.x + rect.width / 2, rect.y + rect.height / 2); // Сохранение центра
            histograms[new_id] = compute_histogram(image, rect); // Вычисление гистограммы для нового объекта
            // Добавление нового объекта в результирующий вектор
            objects_bbs_ids.push_back({ rect.x, rect.y, rect.width, rect.height, new_id });
        }
    }

    return objects_bbs_ids; // Возврат обновленного списка объектов
}

// Метод для возврата сохраненных гистограмм
std::unordered_map<int, cv::Mat> HungarianTracker::retain_histograms() {
    return histograms; // Возврат текущих гистограмм объектов
}