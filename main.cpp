#include <GL/glut.h>
//#include <vector>
#include <iostream>
//#include <fstream>
//#include <cmath>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//#include <math.h>
//#include <windows.h>
//#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#define PI 3.14159f

using namespace std;
void draw();

// To use the dedicated gpu on my laptop
extern "C" {
	_declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}

/* Global */
bool zoom_plus = false, zoom_minus = false, x_plus = false, x_minus = false, y_plus = false, y_minus = false;

int last_mx = 0, last_my = 0, cur_mx = 0, cur_my = 0;
int arcball_on = false;
int screen_width = 500;
int screen_height = 500;

int verts_per_face = 53490;
cv::Mat shapeMU; // (160470, 1) - Mean face Vertices
cv::Mat texMU; // (160470, 1) - Mean face texture

cv::Mat tl; // (106466, 3) - Triangulation Data

cv::Mat shapeEV; // (199, 1) - EigenValues Vertices
cv::Mat texEV; // (199, 1) - Eigenvalue Texture

cv::Mat shapePC; // (160470, 199) - Principal Componenets Vertices
cv::Mat texPC; // (160470, 199) - Principal Components Texture

cv::Mat img(screen_height, screen_width, CV_8UC4);
bool screenshot = false, run_gd_iteration = false;
cv::Mat in_put(screen_height, screen_width, CV_8UC4);
cv::Mat sobel_deriv_dx, sobel_deriv_dy;


cv::Mat  shapeMorphableModel;
cv::Mat  texMorphableModel;

bool both_opaque = true;
cv::Mat alpha_vec(199, 1, CV_32F);
cv::Mat beta_vec(199, 1, CV_32F);
float var = 0.5f;

cv::Point3f centroid;

float inf = std::numeric_limits<float>::infinity();
float minx = inf, miny = inf, minz = inf;
float maxx = -1 * inf, maxy = -1 * inf, maxz = -1 * inf;

float alpha_blend = 0.5;

// Global Model matrix : an identity matrix (model will be at the origin)
glm::mat4 Model;
// Global Camera matrix
glm::mat4 View;

//glm::mat4 View = glm::lookAt(
//	glm::vec3(5,5,0),	// the position of your camera, in world space
//	glm::vec3(0,0,3),	// where you want to look at, in world space
//	glm::vec3(0, 1, 0)		// probably glm::vec3(0,1,0), but (0,-1,0) would make you looking upside-down, which can be great too
//);

/* process menu option 'op' */
void menu(int op) {
	switch (op) {
	case 'Q':
	case 'q':
		exit(0);
	}
}

// Callback for when program is idle
void idle() {
	float scale_factor = 5000;
	if (zoom_plus) {
		screenshot = true;
		View = glm::translate(View, glm::vec3(0, 0, scale_factor));
		glutPostRedisplay();
	}
	else if (zoom_minus) {
		screenshot = true;
		View = glm::translate(View, glm::vec3(0, 0, -scale_factor));
		glutPostRedisplay();
	}
	else if (x_plus) {
		screenshot = true;
		View = glm::translate(View, glm::vec3(scale_factor, 0, 0));
		glutPostRedisplay();
	}
	else if (x_minus) {
		screenshot = true;
		View = glm::translate(View, glm::vec3(-scale_factor, 0, 0));
		glutPostRedisplay();
	}
	else if (y_plus) {
		screenshot = true;
		View = glm::translate(View, glm::vec3(0, scale_factor, 0));
		glutPostRedisplay();
	}
	else if (y_minus) {
		screenshot = true;
		View = glm::translate(View, glm::vec3(0, -scale_factor, 0));
		glutPostRedisplay();
	}
}

// Creates Morphable Model as linear combo of the principle components
void generate_new_face() {
	/*cv::Mat shp_std_devs;
	cv::sqrt(shapeEV, shp_std_devs);*/
	cv::Mat pc_weighted_combo;
	cv::multiply(shapeEV, alpha_vec, pc_weighted_combo);
	shapeMorphableModel = shapeMU + shapePC * pc_weighted_combo;

	/*cv::Mat tex_std_devs;
	cv::sqrt(texEV, tex_std_devs);*/
	cv::Mat pc_weighted_combo_tex;
	cv::multiply(texEV, beta_vec, pc_weighted_combo_tex);
	texMorphableModel = texMU + texPC * pc_weighted_combo_tex;
}

/* executed when a regular key is pressed */
void keyboardDown(unsigned char key, int x, int y) {
	screenshot = true;
	switch (key) {
	case 'Q':
	case 'q':
	case  27:   // ESC
		exit(0);
	case '+':
	case '=':
		zoom_plus = true;
		zoom_minus = false;
		break;
	case '-':
	case '_':
		zoom_minus = true;
		zoom_plus = false;
		break;
	case 'd':
	case 'D':
		x_plus = true;
		x_minus = false;
		break;
	//case 'x':
	//case 'X':
	//	break;
	case 'a':
	case 'A':
		x_minus = true;
		x_plus = false;
		break;
	case 'w':
	case 'W':
		y_plus = true;
		y_minus = false;
		break;
	case 's':
	case 'S':
		y_minus = true;
		y_plus = false;
		break;
	case '5':
		cv::randn(alpha_vec, cv::Scalar(0), cv::Scalar(var));
		cv::randn(beta_vec, cv::Scalar(0), cv::Scalar(var));
		generate_new_face();
		break;
	case '8':
		var += 0.25f;
		cout << "variance is " << var << "\n";
		break;
	case '2':
		var -= 0.25f;
		cout << "variance is " << var << "\n";
		break;
	case '4':
		alpha_blend += .1f;
		break;
	case '6':
		alpha_blend -= .1f;
		break;
	case '0':
		both_opaque = !both_opaque;
		cout << "both opaque: " << both_opaque << "\n";
		break;
	case 'z':
	case 'Z':
		run_gd_iteration = true;
		break;
		//case 'r':
		//case 'R':
		//	zoom_minus = false;
		//	zoom_plus = false;
		//	zoom = 0;
		//	View[3] = glm::vec4(0, 0, zoom, 1);
		//	glutPostRedisplay();
		//	break;
	}
	glutPostRedisplay();
}

/* executed when a regular key is released */
void keyboardUp(unsigned char key, int x, int y) {
	switch (key) {
	case '+':
	case '=':
		zoom_plus = false;
		break;
	case '-':
	case '_':
		zoom_minus = false;
		break;
	case 'd':
	case 'D':
		x_plus = false;
		x_minus = false;
		break;
	case 'a':
	case 'A':
		x_minus = false;
		x_plus = false;
		break;
	case 'w':
	case 'W':
		y_plus = false;
		y_minus = false;
		break;
	case 's':
	case 'S':
		y_minus = false;
		y_plus = false;
		break;
	}
}

/* executed when a special key is pressed */
void keyboardSpecialDown(int k, int x, int y) {

}

/* executed when a special key is released */
void keyboardSpecialUp(int k, int x, int y) {

}

/**
* Get a normalized vector from the center of the virtual ball O to a
* point P on the virtual ball surface, such that P is aligned on
* screen's (X,Y) coordinates.  If (X,Y) is too far away from the
* sphere, return the nearest point on the virtual ball surface.
*/
glm::vec3 get_arcball_vector(int x, int y) {
	glm::vec3 P = glm::vec3(1.0*x / screen_width * 2 - 1.0, 1.0*y / screen_height * 2 - 1.0, 0);
	P.y = -P.y;
	float OP_squared = P.x * P.x + P.y * P.y;
	if (OP_squared <= 1 * 1)
		P.z = sqrt(1 * 1 - OP_squared);  // Pythagore
	else
		P = glm::normalize(P);  // nearest point
	return P;
}

/* reshaped window */
void reshape(int width, int height) {
	//GLfloat fieldOfView = 90.0f;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	float r = sqrt((maxx - centroid.x)*(maxx - centroid.x) + (maxy - centroid.y)*(maxy - centroid.y) + (maxz - centroid.z)*(maxz - centroid.z));
	float fDistance = r / tanf(PI / 6); // where 0.57735f is tan(30 degrees)
	double dNear = fDistance - r;
	double dFar = fDistance + r;
	glFrustum(-r, +r, -r, +r, dNear, dFar);
	//gluPerspective(fieldOfView, (GLfloat)width / (GLfloat)height, 0.1, 500.0);
	Model = glm::mat4(1.0f);
	View = glm::lookAt(glm::vec3(0.0f, 0.0f, fDistance), glm::vec3(centroid.x, centroid.y, centroid.z), glm::vec3(0.0f, 1.0f, 0.0f));
	// remember to use this to translate the model, (via the view matrix to preserve rotation stuff)
	View = glm::translate(View, glm::vec3(-centroid.x, -centroid.y, -centroid.z));

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(glm::value_ptr(View*Model));
}

void onMouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		screenshot = true;
		arcball_on = true;
		last_mx = cur_mx = x;
		last_my = cur_my = y;
		cout << "click location: " << x << ", " << y << "\n";
	}
	else {
		arcball_on = false;
	}
}

/* render the scene */
void draw() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(glm::value_ptr(View * Model));

	// Push the Modelview matrix, so we can do weird translations and what not if neccesary, as long as we do't change model or view though
	// it should not matter, since we are manually controlling these matrices :)
	//glPushMatrix();
	//glLoadMatrixf(glm::value_ptr(View * glm::translate(Model, glm::vec3(0, 0, 0))));
	glBegin(GL_TRIANGLES);
	for (int i = 0; i < tl.rows; i++) {
		int xvert1 = (int)(tl.at<unsigned short>(i, 0) - 1) * 3;
		int xvert2 = (int)(tl.at<unsigned short>(i, 1) - 1) * 3;
		int xvert3 = (int)(tl.at<unsigned short>(i, 2) - 1) * 3;


		// This is the mean face, drawing for reference to see how it compares to the other one
		/*glPushMatrix();
			glTranslatef(maxx - minx, 0, 0);
			glColor3f(texMU.at<float>(xvert1, 0)/255.0, texMU.at<float>(xvert1 + 1, 0) / 255.0, texMU.at<float>(xvert1 + 2, 0) / 255.0);
			glVertex3f(shapeMU.at<float>(xvert1, 0), shapeMU.at<float>(xvert1 + 1, 0), shapeMU.at<float>(xvert1 + 2, 0));

			glColor3f(texMU.at<float>(xvert2, 0) / 255.0, texMU.at<float>(xvert2 + 1, 0) / 255.0, texMU.at<float>(xvert2 + 2, 0) / 255.0);
			glVertex3f(shapeMU.at<float>(xvert2, 0), shapeMU.at<float>(xvert2 + 1, 0), shapeMU.at<float>(xvert2 + 2, 0));

			glColor3f(texMU.at<float>(xvert3, 0) / 255.0, texMU.at<float>(xvert3 + 1, 0) / 255.0, texMU.at<float>(xvert3 + 2, 0) / 255.0);
			glVertex3f(shapeMU.at<float>(xvert3, 0), shapeMU.at<float>(xvert3 + 1, 0), shapeMU.at<float>(xvert3 + 2, 0));
		glPopMatrix();*/

		// This is the Morphable Model Face, which adds weights to the principal components of the data matrix
		glm::vec3 p1(shapeMorphableModel.at<float>(xvert1, 0), shapeMorphableModel.at<float>(xvert1 + 1, 0), shapeMorphableModel.at<float>(xvert1 + 2, 0));
		glm::vec3 p2(shapeMorphableModel.at<float>(xvert2, 0), shapeMorphableModel.at<float>(xvert2 + 1, 0), shapeMorphableModel.at<float>(xvert2 + 2, 0));
		glm::vec3 p3(shapeMorphableModel.at<float>(xvert3, 0), shapeMorphableModel.at<float>(xvert3 + 1, 0), shapeMorphableModel.at<float>(xvert3 + 2, 0));

		glm::vec3 N = glm::normalize(glm::cross(p2 - p1, p3 - p1));

		glColor3f(texMorphableModel.at<float>(xvert1, 0) / 255.0, texMorphableModel.at<float>(xvert1 + 1, 0) / 255.0, texMorphableModel.at<float>(xvert1 + 2, 0) / 255.0);
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		glNormal3f(N.x, N.y, N.z);
		glVertex3f(p1.x, p1.y, p1.z);

		glColor3f(texMorphableModel.at<float>(xvert2, 0) / 255.0, texMorphableModel.at<float>(xvert2 + 1, 0) / 255.0, texMorphableModel.at<float>(xvert2 + 2, 0) / 255.0);
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		glNormal3f(N.x, N.y, N.z);
		glVertex3f(p2.x, p2.y, p2.z);

		glColor3f(texMorphableModel.at<float>(xvert3, 0) / 255.0, texMorphableModel.at<float>(xvert3 + 1, 0) / 255.0, texMorphableModel.at<float>(xvert3 + 2, 0) / 255.0);
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		glNormal3f(N.x, N.y, N.z);
		glVertex3f(p3.x, p3.y, p3.z);
	}

	glEnd();

	// Display the image of face overlayed with the 3d morphable model
	if (screenshot == true) {
		screenshot = false;

		glReadPixels(0, 0, img.cols, img.rows, GL_BGRA_EXT, GL_UNSIGNED_BYTE, img.data);
		cv::flip(img, img, 0);

		cv::Mat overlayed_image;

		if(!both_opaque)
			cv::addWeighted(in_put, alpha_blend, img, 1.0 - alpha_blend, 0, overlayed_image);
		else
			cv::addWeighted(in_put, .7, img, 1.0, 0, overlayed_image);

		cv::imshow("Display window", overlayed_image);
	}
	
	// Run an iteration of gradient descent... should be until a re-rendering of the 3d model image. happens once every 1000 coefficient updates.
	if(run_gd_iteration == true){
		run_gd_iteration = false;
		cout << "running grad descent iteration(1000 updates)...\n";
		cout << "calculating the 2d image projections of 3d image:\n";

		glm::mat4 proj;
		glGetFloatv(GL_PROJECTION_MATRIX, glm::value_ptr(proj));

		const int num_triangles_select = 40;
		for (int grad_descent_iteration = 0; grad_descent_iteration < 1; grad_descent_iteration++) {

			// Pick Random Triangles
			int rand_triangles[num_triangles_select];
			for (int i = 0; i < num_triangles_select; i++) {
				int random_num = rand() % tl.rows;
				rand_triangles[i] = random_num;
				cout << random_num << ", ";
			}

			// Draw selected Triangles Green just to make sure the right thing is happening
			for (int i = 0; i < num_triangles_select; i++) {
				int xvert1 = (int)(tl.at<unsigned short>(rand_triangles[i], 0) - 1) * 3;
				int xvert2 = (int)(tl.at<unsigned short>(rand_triangles[i], 1) - 1) * 3;
				int xvert3 = (int)(tl.at<unsigned short>(rand_triangles[i], 2) - 1) * 3;

				// This is the Morphable Model Face, which adds weights to the principal components of the data matrix
				glm::vec3 p1(shapeMorphableModel.at<float>(xvert1, 0), shapeMorphableModel.at<float>(xvert1 + 1, 0), shapeMorphableModel.at<float>(xvert1 + 2, 0));
				glm::vec3 p2(shapeMorphableModel.at<float>(xvert2, 0), shapeMorphableModel.at<float>(xvert2 + 1, 0), shapeMorphableModel.at<float>(xvert2 + 2, 0));
				glm::vec3 p3(shapeMorphableModel.at<float>(xvert3, 0), shapeMorphableModel.at<float>(xvert3 + 1, 0), shapeMorphableModel.at<float>(xvert3 + 2, 0));

				glm::vec3 N = glm::normalize(glm::cross(p2 - p1, p3 - p1));
				glm::vec3 p_ctr = (p1 + p2 + p3) / 3.0f;

				glColor3f(0, 1, 0);

				glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
				glNormal3f(N.x, N.y, N.z);
				glVertex3f(p1.x, p1.y, p1.z);

				glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
				glNormal3f(N.x, N.y, N.z);
				glVertex3f(p2.x, p2.y, p2.z);

				glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
				glNormal3f(N.x, N.y, N.z);
				glVertex3f(p3.x, p3.y, p3.z);
			}

			// Loop through the random triangles and perform update the sum based on the derivative for each one
			for (int i = 0; i < num_triangles_select; i++) {
				int xvert1 = (int)(tl.at<unsigned short>(rand_triangles[i], 0) - 1) * 3;
				int xvert2 = (int)(tl.at<unsigned short>(rand_triangles[i], 1) - 1) * 3;
				int xvert3 = (int)(tl.at<unsigned short>(rand_triangles[i], 2) - 1) * 3;

				// This is the Morphable Model Face, which adds weights to the principal components of the data matrix
				glm::vec3 p1(shapeMorphableModel.at<float>(xvert1, 0), shapeMorphableModel.at<float>(xvert1 + 1, 0), shapeMorphableModel.at<float>(xvert1 + 2, 0));
				glm::vec3 p2(shapeMorphableModel.at<float>(xvert2, 0), shapeMorphableModel.at<float>(xvert2 + 1, 0), shapeMorphableModel.at<float>(xvert2 + 2, 0));
				glm::vec3 p3(shapeMorphableModel.at<float>(xvert3, 0), shapeMorphableModel.at<float>(xvert3 + 1, 0), shapeMorphableModel.at<float>(xvert3 + 2, 0));

				glm::vec3 N = glm::normalize(glm::cross(p2 - p1, p3 - p1));
				glm::vec3 p_ctr = (p1 + p2 + p3) / 3.0f;				

				glm::vec4 image_coord = (glm::mat4(proj) * (glm::mat4(View) * (glm::mat4(Model) * glm::vec4(p_ctr, 1.0f))));
				glm::vec2 img_coord_twoD = glm::vec2(image_coord) / image_coord.w;
				float wdiv = screen_width / 2;
				float hdiv = screen_height / 2;

				img_coord_twoD.x = round(img_coord_twoD.x * wdiv + wdiv);
				img_coord_twoD.y = round(-img_coord_twoD.y * hdiv + hdiv);
				//cout << "image coord for the triangle pt 1 is: " << "(" << img_coord_twoD.x << ", " << img_coord_twoD.y << ", " << image_coord.z << ")\n";
				//cout << "projective coord pt 1 is: " << "(" << image_coord.x << ", " << image_coord.y << ", " << image_coord.z << ")\n";
				//in_put 2 * ||Input - Projected Model Image|| * [(d_In / d_s) - (d_Projected / d_s)] * v_j

				// Get derivs of the image w/r/t input image at projected triangle_coords
				float input_img_dx = sobel_deriv_dx.at<float>(img_coord_twoD.x, img_coord_twoD.y);
				float input_img_dy = sobel_deriv_dx.at<float>(img_coord_twoD.x, img_coord_twoD.y);

				// The deriv of the
				//partial_wrt_input * glm::mat4(proj) * glm::mat4(View) * glm::mat4(Model);
			}
		}

	}
	//glPopMatrix();

	glFlush();
	glutSwapBuffers();
}

void onMotion(int x, int y) {
	if (arcball_on) {  // if left button is pressed
		cur_mx = x;
		cur_my = y;

		if (cur_mx != last_mx || cur_my != last_my) {
			glm::vec3 va = get_arcball_vector(last_mx, last_my);
			glm::vec3 vb = get_arcball_vector(cur_mx, cur_my);
			float angle = acos(min(1.0f, glm::dot(va, vb))) / 45;
			glm::vec3 axis_in_camera_coord = glm::cross(va, vb);
			glm::mat3 camera2object = glm::inverse(View*Model);
			glm::vec3 axis_in_object_coord = camera2object * axis_in_camera_coord;
			Model = glm::rotate(Model, glm::degrees(angle), axis_in_object_coord);
			last_mx = cur_mx;
			last_my = cur_my;
			glutPostRedisplay();
		}
	}
}

/* initialize OpenGL settings */
void initGL(int width, int height) {
	reshape(width, height);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1.0f);

	GLfloat light_ambient[] =
	{ 0.5, 0.5, 0.5, 1.0 };
	GLfloat light_diffuse[] =
	{ 1.0, 1.0, 1.0, 1.0 };
	GLfloat light_specular[] =
	{ 1.0, 1.0, 1.0, 1.0 };
	GLfloat light_position[] =
	{ 0.0, 0.0, 1.0, 0.0 };

	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	//glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_COLOR_MATERIAL);

	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);

	glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
	glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());
}

// Function used from http://stackoverflow.com/a/17820615
string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

void load_bfm_data() {
	/*for (int i = 0; i < 199; i++)	{
		if (i >= 0) {
			alpha_vec.at<float>(i, 0) = 0.0f;
			beta_vec.at<float>(i, 0) = 0.0f;
		}else{
			alpha_vec.at<float>(i, 0) = 2.0f;
			beta_vec.at<float>(i, 0) = 2.0f;
		}
	}*/

	cv::randn(alpha_vec, cv::Scalar(0), cv::Scalar(var));
	cv::randn(beta_vec, cv::Scalar(0), cv::Scalar(var));
	//cv::Mat dest_vec;

	//cout << "testvec is = " << testvec.t() << "\n\n";
	/*cout << "alpha_vec is = " << alpha_vec.t() << "\n\n";
	cv::sqrt(alpha_vec, alpha_vec);
	cout << "alpha_vec is = " << alpha_vec.t() << "\n\n";*/

	//cout << "alpha_vec shape is = " << alpha_vec << "\n\n";
	cout << "alpha_vec shape is = " << alpha_vec.size() << "\n\n";
	cout << "beta_vec  shape is = " << beta_vec.size() << "\n\n";

	//cout << "alpha_vec is = " << alpha_vec.t() << "\n\n";
	//cout << "beta_vec  is = " << beta_vec.t() << "\n\n";

	/*cout << "sum is = " << cv::sum(alpha_vec) << "\n";
	cout << "beta_vec sum is = " << cv::sum(beta_vec) << "\n\n";
	cv::Mat1d newVec = cv::sum(alpha_vec)[0] * alpha_vec;
	cout << "after division, it is = " << cv::sum(newVec)[0] << "\n\n";
	cv::sum(beta_vec);*/

	//printf("OpenCV: %s", cv::getBuildInformation().c_str());
	cout << "Loading Mean Shape Data...\n";
	cv::FileStorage("C:/Users/utsav/Documents/3D Morphable Model Project/yaml/shapeMU.yaml", cv::FileStorage::FORMAT_YAML)["shapeMU"] >> shapeMU;
	cv::FileStorage("C:/Users/utsav/Documents/3D Morphable Model Project/yaml/texMU.yaml", cv::FileStorage::FORMAT_YAML)["texMU"] >> texMU;

	cout << "Loading Triangulation Data...\n";
	cv::FileStorage("C:/Users/utsav/Documents/3D Morphable Model Project/yaml/tl.yaml", cv::FileStorage::FORMAT_YAML)["tl"] >> tl; // (106466, 3) - Triangulation Data

	cout << "Loading Eigenvalue Data...\n";
	cv::FileStorage("C:/Users/utsav/Documents/3D Morphable Model Project/yaml/shapeEV.yaml", cv::FileStorage::FORMAT_YAML)["shapeEV"] >> shapeEV; // (199, 1) - EigenValues Vertices
	cv::FileStorage("C:/Users/utsav/Documents/3D Morphable Model Project/yaml/texEV.yaml", cv::FileStorage::FORMAT_YAML)["texEV"] >> texEV; // (199, 1) - Eigenvalue Texture

	cout << "Loading Prcinicpal Component Data...\n";
	cv::FileStorage("C:/Users/utsav/Documents/3D Morphable Model Project/yaml/shapePC.yaml", cv::FileStorage::FORMAT_YAML)["shapePC"] >> shapePC; // (160470, 199) - Principal Componenets Vertices
	cv::FileStorage("C:/Users/utsav/Documents/3D Morphable Model Project/yaml/texPC.yaml", cv::FileStorage::FORMAT_YAML)["texPC"] >> texPC; // (160470, 199) - Principal Components Texture 
	cout << "Done...\n";

	float sumx = 0, sumy = 0, sumz = 0;
	//sanity checks
	assert(100000.0f < inf);
	assert(-100000.0f > -1 * inf);

	for (int i = 0; i < verts_per_face; i += 3) {
		float x = shapeMU.at<float>(i, 0);
		float y = shapeMU.at<float>(i + 1, 0);
		float z = shapeMU.at<float>(i + 2, 0);

		sumx += x;
		sumy += y;
		sumz += z;

		if (x < minx)
			minx = x;
		if (y < miny)
			miny = y;
		if (z < minz)
			minz = z;

		if (x > maxx)
			maxx = x;
		if (y > maxy)
			maxy = y;
		if (z > maxz)
			maxz = z;
	}

	// Calculate the center of point mesh
	centroid = cv::Point3f(sumx / verts_per_face, sumy / verts_per_face, sumz / verts_per_face);
	generate_new_face();
}

/* initialize GLUT settings, register callbacks, enter main loop */
int main(int argc, char** argv) {
	cout << "Loading input face...\n";
	in_put = cv::imread("C:/Users/utsav/Documents/3D Morphable Model Project/putin2.png", cv::IMREAD_UNCHANGED);
	cv::resize(in_put, in_put, cv::Size(), screen_width / 1500.0f, screen_height / 1500.0f);

	cv::Sobel(in_put, sobel_deriv_dx, CV_32F, 1, 0);
	cv::Sobel(in_put, sobel_deriv_dy, CV_32F, 1, 0);
	//cv::imshow("Putin window", sobel_deriv);
	cout << "sobel_deriv_dx Shape is: " << sobel_deriv_dx.size() << " \n";
	cout << "sobel_deriv_dx Type is: " << type2str(sobel_deriv_dx.type()) << " \n";

	load_bfm_data();

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(screen_width, screen_height);
	//glutInitWindowPosition(100, 100);
	glutCreateWindow("284 Project Start");

	// register glut call backs
	glutKeyboardFunc(keyboardDown);
	glutKeyboardUpFunc(keyboardUp);
	glutSpecialFunc(keyboardSpecialDown);
	glutSpecialUpFunc(keyboardSpecialUp);
	glutIdleFunc(idle);
	glutMouseFunc(onMouse);
	glutMotionFunc(onMotion);
	glutReshapeFunc(reshape);
	glutDisplayFunc(draw);

	glutIgnoreKeyRepeat(true); // ignore keys held down

	int subMenu = glutCreateMenu(menu);	// create a sub menu
	glutAddMenuEntry("Do nothing", 0);
	glutAddMenuEntry("Really Quit", 'q');

	// create main "right click" menu
	glutCreateMenu(menu);
	glutAddSubMenu("Sub Menu", subMenu);
	glutAddMenuEntry("Quit", 'q');
	glutAttachMenu(GLUT_RIGHT_BUTTON);

	initGL(screen_width, screen_height);
	glutMainLoop();
	return 0;
}