#include <opencv2/aruco.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "aruco_samples_utility.hpp";
#include <queue> 
#include <algorithm> 
#include <map>
#include <opencv2/core/mat.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "camera.h"

float Polygon_Area(std::vector<cv::Point2f>& list_of_points)
{
	float result{ 0.f };
	int size_list{ 0 }, i{ 0 };
	size_list = static_cast<int>(list_of_points.size());

	for (i = 0; i < (size_list - 1); i++)
	{
		result += (list_of_points[i].x * list_of_points[i + 1].y -
			list_of_points[i + 1].x * list_of_points[i].y);
	}

	result += (list_of_points[i].x * list_of_points[0].y -
		list_of_points[0].x * list_of_points[i].y);

	result *= 0.5f;

	return result;
}


std::map<int, cv::Mat> rot;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 10.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
bool pressedSolve = false;
// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

std::vector<glm::vec3> cubePosition;
std::queue<glm::vec3> cubePositionTemp;

int main() {

	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	
	Shader program("vertexShader.glsl", "fragmentShader.glsl");
	
	float vertices[] = {
		-0.5f, -0.5f, -0.5f, 
		 0.5f, -0.5f, -0.5f,  
		 0.5f,  0.5f, -0.5f,  
		 0.5f,  0.5f, -0.5f,  
		-0.5f,  0.5f, -0.5f,  
		-0.5f, -0.5f, -0.5f, 

		-0.5f, -0.5f,  0.5f,  
		 0.5f, -0.5f,  0.5f,  
		 0.5f,  0.5f,  0.5f,  
		 0.5f,  0.5f,  0.5f,  
		-0.5f,  0.5f,  0.5f,  
		-0.5f, -0.5f,  0.5f,  

		-0.5f,  0.5f,  0.5f, 
		-0.5f,  0.5f, -0.5f,  
		-0.5f, -0.5f, -0.5f, 
		-0.5f, -0.5f, -0.5f,  
		-0.5f, -0.5f,  0.5f,  
		-0.5f,  0.5f,  0.5f,  

		 0.5f,  0.5f,  0.5f,  
		 0.5f,  0.5f, -0.5f,  
		 0.5f, -0.5f, -0.5f, 
		 0.5f, -0.5f, -0.5f,  
		 0.5f, -0.5f,  0.5f, 
		 0.5f,  0.5f,  0.5f,  

		-0.5f, -0.5f, -0.5f,  
		 0.5f, -0.5f, -0.5f, 
		 0.5f, -0.5f,  0.5f, 
		 0.5f, -0.5f,  0.5f, 
		-0.5f, -0.5f,  0.5f,  
		-0.5f, -0.5f, -0.5f, 

		-0.5f,  0.5f, -0.5f,  
		 0.5f,  0.5f, -0.5f,  
		 0.5f,  0.5f,  0.5f, 
		 0.5f,  0.5f,  0.5f,  
		-0.5f,  0.5f,  0.5f, 
		-0.5f,  0.5f, -0.5f
	};

	unsigned int VBO, VAO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	//glBindVertexArray(0);



	std::vector<cv::Scalar> mColorList;
	mColorList.push_back(cv::Scalar(0, 0, 255));
	mColorList.push_back(cv::Scalar(255, 0, 0));
	mColorList.push_back(cv::Scalar(255, 255, 0));
	mColorList.push_back(cv::Scalar(0, 255, 255));
	
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
	rot[4].at<double>(0, 0) = -1;
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

	std::cout << "OpenCV version : " << CV_VERSION << std::endl;
	

	cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
	cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
	cv::aruco::ArucoDetector detector(dictionary, detectorParams);

	std::vector<std::vector<cv::Point2f>> markerCorners;
	std::vector<int> markerIds;
	std::vector<float> markerAreas;
	
	glLineWidth(10.0f);
	while (inputVideo.grab()) {

		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		//gray = cv2.cvtcolor(imageCopy,cv2.color_bgr2gray)
		cv::Mat image, imageCopy;
		inputVideo.retrieve(image);
		image.copyTo(imageCopy);
		
		detector.detectMarkers(imageCopy, markerCorners, markerIds);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		program.use();

		// camera/view transformation
		glm::mat4 view = camera.GetViewMatrix();
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

		program.setMat4("projection", projection);
		program.setMat4("view", view);

		// render box
		glBindVertexArray(VAO);
		if (markerCorners.size() > 0) {

			int circle_size{ 4 };
			for (int i = 0; i < markerCorners.size(); i++)
			{
				for (int j = 0; j < markerCorners[i].size(); j++)
				{
					cv::circle(
						imageCopy, markerCorners[i][j],
						circle_size, mColorList[j], cv::FILLED);
				}

				markerAreas.push_back(Polygon_Area(markerCorners[i]));
				//std::cout << "\nArea of marker id (" << markerIds[i] << " )" << markerAreas[i];
			}

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
			
			
			std::vector<cv::Point2f> imgpt;
			cv::projectPoints(centPoints.ptr<cv::Vec3f>(0)[minId], computed_rvec, tvecs, cameraMatrix, distCoeffs, imgpt);
			
			for (const auto& point : imgpt) {
				cv::circle(imageCopy, point, 5, cv::Scalar(255, 0, 255), cv::FILLED);
			}

			// Normalize points to the range [-1, 1]
			
			// Calculate the normalization factors
			//TODO: calculate the max and min of the camera window		
			
			float scaleX = 2.0f / (500 - 20);
			float scaleY = 2.0f / (500 - 20);

			// Normalize the points
			for (auto& point : imgpt) {
				point.x = scaleX * (point.x - 20) - 1.0f;
				point.y = (scaleY * (point.y - 20) - 1.0f) * -1.0f;
			}
			
			
			// Update the vertex data for the center point
			//only has x y coordinates
			glm::vec3 centerr(imgpt[0].x, imgpt[0].y, 0.0f);
			
			cubePositionTemp.push(centerr);

			glm::mat4 model = glm::mat4(1.0f);
			model = glm::translate(model, centerr);
			program.setMat4("model", model);
			glDrawArrays(GL_TRIANGLES, 0, 36);
		}
		
		processInput(window);

		if (!cubePosition.empty()) {
			for (auto i : cubePosition) {
				glm::mat4 model = glm::mat4(1.0f);
				model = glm::translate(model, i);
				program.setMat4("model", model);
				glDrawArrays(GL_TRIANGLES, 0, 36);
			}
		}

		glfwSwapBuffers(window);
		glfwPollEvents();

		cv::imshow("Out", imageCopy);
		int key = cv::waitKey(20);
		if (key == 'q')
		{
			std::cout << "q key is pressed by the user. Stopping the video\n";
			break;
		}
		if (!cubePositionTemp.empty()) {
			cubePositionTemp.pop();
		}
	}
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);

	glfwTerminate();
	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	//move the camara
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, deltaTime);

	//save location and draw it
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		cubePosition.push_back(cubePositionTemp.front());
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}