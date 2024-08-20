#include <opencv2/opencv.hpp>
#include <thread>
#include <iostream>
#include "HungarianTracker.h"

// Функция обработки видеофайла
void processing(const std::string& file, const std::string& file_name) {
    cv::VideoCapture cap(file); // Открываем видеопоток
    if (!cap.isOpened()) { // Проверяем, удалось ли открыть файл
        std::cerr << "Error: Cannot open video file " << file << std::endl; // Сообщаем об ошибке
        return;
    }

    const double alpha = 0.999;
    bool isFirstTime = true;      // Флаг для определения первого кадра
    HungarianTracker tracker;      // Инициализируем трекер объектов

    cv::Mat bg_img; // Матрица для хранения фона

    while (true) { // Начинаем бесконечный цикл обработки кадров
        cv::Mat frame; // Матрица для текущего кадра
        if (!cap.read(frame)) { // Читаем текущий кадр
            std::cout << "End of video stream " << file << std::endl; // Сообщаем о завершении видео
            break; // Выход из цикла
        }

        cv::resize(frame, frame, cv::Size(), 0.5, 0.5); // Уменьшаем размер кадра до 50%

        cv::Mat frame_copy = frame.clone(); // Клонируем текущий кадр
        frame_copy(cv::Range(350, 360), cv::Range(340, 380)) = cv::Scalar(127, 127, 127); // Заливаем область серым цветом

        if (isFirstTime) { // Если это первый кадр
            bg_img = frame_copy; // Устанавливаем фон
            isFirstTime = false; // Устанавливаем флаг, что первый кадр обработан
        }
        else {
            // Смешиваем текущее изображение с фоном
            cv::addWeighted(frame_copy, 1 - alpha, bg_img, alpha, 0, bg_img);
        }

        cv::Mat fg_img; // Матрица для изображения с объектами
        cv::absdiff(frame_copy, bg_img, fg_img); // Вычисляем разность между текущим кадром и фоном
        cv::Mat gray; // Матрица для изображения в градациях серого
        cv::cvtColor(fg_img, gray, cv::COLOR_BGR2GRAY); // Преобразуем в оттенки серого
        cv::Mat thresh; // Матрица для бинарного изображения
        cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 51, 8); // Применяем адаптивный порог

        std::vector<std::vector<cv::Point>> contours; // Вектор для хранения контуров
        cv::findContours(thresh, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE); // Находим контуры на бинарном изображении

        std::vector<cv::Rect> detections; // Вектор для хранения обнаруженных объектов

        if (!contours.empty()) { // Проверяем, есть ли найденные контуры
            // Находим самый большой контур
            auto largest_contour = *std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) < cv::contourArea(b); // Сравниваем площадь контуров
                }
            );
            cv::Rect bounding_box = cv::boundingRect(largest_contour); // Получаем ограничивающую рамку этого контура
            detections.push_back(bounding_box); // Сохраняем рамку в вектор
        }

        auto boxes_ids = tracker.update(detections, frame_copy); // Обновляем трекер и получаем рамки с ID объектов

        // Обрабатываем результаты трекера
        for (const auto& box_id : boxes_ids) {
            int x = box_id[0]; // Координата x
            int y = box_id[1]; // Координата y
            int w = box_id[2]; // Ширина рамки
            int h = box_id[3]; // Высота рамки
            int id = box_id[4]; // ID объекта

            // Отображаем ID объекта на кадре
            cv::putText(frame, std::to_string(id), cv::Point(x, y - 15),
                cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0), 2);
            // Рисуем рамку вокруг объекта
            cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 0, 255), 3);
        }

        cv::imshow(file_name, frame); // Отображаем кадр в окне
        if (cv::waitKey(15) == 'q') { // Проверяем нажатие клавиши 'q' для выхода
            break; // Выход из цикла
        }
    }
    cap.release(); // Освобождаем видеопоток
    cv::destroyAllWindows(); // Закрываем все окна
}

// Функция для обработки видео с первой камеры
void first_cam() {
    processing("D:/3.Camera 2017-05-29 16-23-04_137 [3m3s].avi", "First Camera"); // Запускаем обработку
}

// Функция для обработки видео со второй камеры
void second_cam() {
    processing("D:/4.Camera 2017-05-29 16-23-04_137 [3m3s].avi", "Second Camera"); // Запускаем обработку
}

// Главная функция
int main() {
    std::thread first_thread(first_cam); // Создаем поток для первой камеры
    std::thread second_thread(second_cam); // Создаем поток для второй камеры

    first_thread.join(); // Ожидаем завершения потока первой камеры
    second_thread.join(); // Ожидаем завершения потока второй камеры

    return 0; // Завершаем программу
}