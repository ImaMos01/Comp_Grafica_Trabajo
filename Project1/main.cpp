#include <opencv2/aruco.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "aruco_samples_utility.hpp";
#include <algorithm> 
#include <map>
#include <opencv2/core/mat.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


std::map<int, cv::Mat> rot;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"	gl_PointSize = 30.0f;\n"
"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
"}\n\0";

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

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	// build and compile our shader program
   // ------------------------------------
   // vertex shader
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	glEnable(GL_PROGRAM_POINT_SIZE);
	// check for shader compile errors
	int success;
	char infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	// fragment shader
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);
	// check for shader compile errors
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	// link shaders
	unsigned int shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	// check for linking errors
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

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

	std::cout << "OpenCV version : " << CV_VERSION << std::endl;
	
	float vertices[] = {
		0.0f, 0.0f, 0.0f,  
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
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	glBindVertexArray(0);


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

			// Calculate the normalization factors
			float scaleX = 2.0f / (600 - 80);
			float scaleY = 2.0f / (500 - 60);

			// Normalize the points
			for (auto& point : imgpt) {
				point.x = scaleX * (point.x - 80) - 1.0f;
				point.y = (scaleY * (point.y - 80) - 1.0f) * -1.0f;
			}

			std::cout << imgpt[0] << "\n";
			// Update the vertex data for the center point
			glm::vec3 centerr(imgpt[0].x, imgpt[0].y, 0.0f);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3), &centerr, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

		}

		glUseProgram(shaderProgram);
		glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
		glDrawArrays(GL_POINTS, 0, 1);
		glBindVertexArray(0);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();

		cv::imshow("Out", imageCopy);
		int key = cv::waitKey(20);
		if (key == 'q')
		{
			std::cout << "q key is pressed by the user. Stopping the video\n";
			break;
		}
	}
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteProgram(shaderProgram);
	glfwTerminate();
	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}