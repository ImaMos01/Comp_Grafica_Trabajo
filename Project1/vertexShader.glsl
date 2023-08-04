#version 330 core
layout (location = 0) in vec3 inPos;

// matrices
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


void main()
{
	gl_Position = projection * view * model * vec4(inPos, 1.0f);
	gl_PointSize = 30.0f;
}