#include <opencv2/aruco.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "aruco_samples_utility.hpp";
#include <algorithm> 
#include <map>
#include <opencv2/core/mat.hpp>


std::map<int, cv::Mat> rot;

int main() {
	rot[0] = cv::Mat::zeros(3, 3, CV_64F);
	rot[0].at<double>(0, 2) = -1;
	rot[0].at<double>(1, 1) = 1;
	rot[0].at<double>(2, 0) = 1;
	rot[1] = cv::Mat::zeros(3, 3, CV_64F);
	rot[1].at<double>(0, 2) = -1;
	rot[1].at<double>(1, 1) = -1;
	rot[1].at<double>(2, 0) = -1;
	rot[2] = cv::Mat::zeros(3, 3, CV_64F);
	rot[2].at<double>(0, 2) = -1;
	rot[2].at<double>(1, 0) = 1;
	rot[2].at<double>(2, 1) = -1;
	rot[3] = cv::Mat::zeros(3, 3, CV_64F);
	rot[3].at<double>(0, 2) = -1;
	rot[3].at<double>(1, 0) = -1;
	rot[3].at<double>(2, 1) = 1;
	rot[4] = cv::Mat::zeros(3, 3, CV_64F);
	rot[4].at<double>(0, 2) = -1;
	rot[4].at<double>(1, 1) = 1;
	rot[4].at<double>(2, 2) = -1;
	rot[5] = cv::Mat::zeros(3, 3, CV_64F);
	rot[5].at<double>(0, 0) = 1;
	rot[5].at<double>(1, 1) = 1;
	rot[5].at<double>(2, 2) = 1;


	/*
	rot[0] = { {0, 0, -1},{0, 1, 0},{1, 0, 0} };
	rot[1] = { {0, 0, -1},{0, -1, 0},{-1, 0, 0} };
	rot[2] = { {0, 0, -1},{1, 0, 0},{0, -1, 0} };
	rot[3] = { {0, 0, -1},{-1, 0, 0},{0, 1, 0} };
	rot[4] = { {0, 0, -1},{0, 1, 0},{0, 0, -1} };
	rot[5] = { {1, 0, 0},{0, 1, 0},{0, 0, 1} };
	*/

	cv::VideoCapture inputVideo;
	inputVideo.open(0);

	cv::Mat cameraMatrix, distCoeffs;
	float markerLength = 0.05;

	//parametros de la camara
	readCameraParameters("tutorial_camera_params.yml",cameraMatrix,distCoeffs);
	
	//coordenadas del sistema
	cv::Mat objPoints(4, 1, CV_32FC3);
	objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
	objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
	objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
	objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);

	//centroide del cubo dada la cara
	cv::Mat centPoints(6, 1, CV_32FC3);
	centPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-0.025, 0, 0);
	centPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(0.025, 0, 0);
	centPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(0, 0.025, 0);
	centPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(0, -0.025, 0);
	centPoints.ptr<cv::Vec3f>(0)[4] = cv::Vec3f(0, 0, 0.025);
	centPoints.ptr<cv::Vec3f>(0)[5] = cv::Vec3f(0, 0, -0.025);

	while (inputVideo.grab()) {

		//gray = cv2.cvtcolor(imageCopy,cv2.color_bgr2gray)
		cv::Mat image, imageCopy, imageCopy2;
		inputVideo.retrieve(image);
		image.copyTo(imageCopy);
		image.copyTo(imageCopy2);

		cv::cvtColor(image,imageCopy2,cv::COLOR_BGR2GRAY);

		cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
		cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
		cv::aruco::ArucoDetector detector(dictionary, detectorParams);

		std::vector<std::vector<cv::Point2f>> markerCorners;
		std::vector<int> markerIds;
		
		detector.detectMarkers(imageCopy, markerCorners, markerIds);

		if (markerCorners.size() > 0) {
			
			int minId = *std::min_element(markerIds.begin(), markerIds.end());
			int minIndex = std::distance(markerIds.begin(), std::min_element(markerIds.begin(), markerIds.end()));

			int nMarkers = markerCorners.size();
			cv::Mat rvecs, tvecs; //vector de rotacion y translación
			cv::Vec3d center;

			//calibrar camara
			solvePnP(objPoints, markerCorners.at(minIndex), cameraMatrix, distCoeffs, rvecs, tvecs);

			cv::Mat computed_rvec,rot_mat;

			Rodrigues(rvecs, rot_mat);
			cv::Mat computed_rot;
			//multiplicación de matrices
			cv::gemm(rot_mat, rot[minId],1.0, cv::Mat(),0.0,computed_rot);
			
			Rodrigues(computed_rot,computed_rvec);
			
			//cv::drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, computed_rvec, tvecs, markerLength * 0.5f);
			
			std::vector<cv::Point2f> imgpt;
			cv::projectPoints(centPoints.ptr<cv::Vec3f>(0)[minId], computed_rvec, tvecs, cameraMatrix, distCoeffs, imgpt);
			
			for (const auto& point : imgpt) {
				cv::circle(imageCopy, point, 5, cv::Scalar(0, 255, 0), cv::FILLED);
			}
			
		}

		cv::imshow("Out", imageCopy);
		int key = cv::waitKey(20);
		if (key == 'q')
		{
			std::cout << "q key is pressed by the user. Stopping the video\n";
			break;
		}
	}
}