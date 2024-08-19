#include <opencv2/opencv.hpp>
#include <thread>
#include <iostream>
#include "HungarianTracker.h"

void processing(const std::string& file, const std::string& file_name) {
    cv::VideoCapture cap(file);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file " << file << std::endl;
        return;
    }

    const double alpha = 0.999;
    bool isFirstTime = true;
    HungarianTracker tracker;

    cv::Mat bg_img;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            std::cout << "End of video stream " << file << std::endl;
            break;
        }

        cv::resize(frame, frame, cv::Size(), 0.5, 0.5);

        cv::Mat frame_copy = frame.clone();
        frame_copy(cv::Range(350, 360), cv::Range(340, 380)) = cv::Scalar(127, 127, 127);

        if (isFirstTime) {
            bg_img = frame_copy;
            isFirstTime = false;
        }
        else {
            cv::addWeighted(frame_copy, 1 - alpha, bg_img, alpha, 0, bg_img);
        }

        cv::Mat fg_img;
        cv::absdiff(frame_copy, bg_img, fg_img);
        cv::Mat gray;
        cv::cvtColor(fg_img, gray, cv::COLOR_BGR2GRAY);
        cv::Mat thresh;
        cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 51, 8);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> detections;

        if (!contours.empty()) {
            auto largest_contour = *std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) < cv::contourArea(b);
                }
            );
            cv::Rect bounding_box = cv::boundingRect(largest_contour);
            detections.push_back(bounding_box);
        }

        auto boxes_ids = tracker.update(detections, frame_copy);

        for (const auto& box_id : boxes_ids) {
            int x = box_id[0];
            int y = box_id[1];
            int w = box_id[2];
            int h = box_id[3];
            int id = box_id[4];

            cv::putText(frame, std::to_string(id), cv::Point(x, y - 15),
                cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0), 2);
            cv::rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 0, 255), 3);
        }

        cv::imshow(file_name, frame);
        if (cv::waitKey(15) == 'q') {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
}

void first_cam() {
    processing("D:/3.Camera 2017-05-29 16-23-04_137 [3m3s].avi", "First Camera");
}

void second_cam() {
    processing("D:/4.Camera 2017-05-29 16-23-04_137 [3m3s].avi", "Second Camera");
}

int main() {
    std::thread first_thread(first_cam);
    std::thread second_thread(second_cam);

    first_thread.join();
    second_thread.join();

    return 0;
}