#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//Windows対応はしない

#include <helper_gl.h>
#include <GL/freeglut.h>

#include <helper_functions.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define REFRESH_DELAY     (50) //ms
#define D_ANGLE (5) //ms

#define NUM_OF_NEURONS (96749)
#define PRE_LOAD (50)
#define TAU (4.0)
#define DT (1.0)
/////////////////////////////////////
//constants
const unsigned int window_width = 1024;
const unsigned int window_height = 1024;

const unsigned int mesh_width    = 256;
const unsigned int mesh_height   = 256;

////! VBO variables
GLuint *vbo;
struct cudaGraphicsResource** vbo_resources;


/////////////////////////////////////
bool initGL(int *argc, char **argv);

void display();
void display2();

void timerEvent(int value);
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

void createVBOs(GLuint **vbo_list, struct cudaGraphicsResource ***vbo_resources, int vbo_num, unsigned int size, unsigned int vbo_res_flags);
void deleteVBOs(GLuint *vbo, struct cudaGraphicsResource **vbo_res, int vbo_num);

/////////////////////////////////////

void LoadPos( float4 *pos, char* type_arr, GLfloat frame_vertex[8][3], char *filename, int num );



///////////////////////////////////// CUDA
__global__ void simple_vbo_kernel(float4 *pos, float4 *col, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;
    float c = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f + 0.5f;

    // write output vertex
    pos[y*width+x] = make_float4(u, w, v, 1.0f);
    col[y*width+x] = make_float4(c, c, c, 1.0f);
}

__global__ void init_act_spike( float *d_activity, char *d_spike, int num ){
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id < num){
        d_activity[id] = 0.f;
        d_spike[id] = 0;
    }
};

__global__ void update_activity(float4 *color, char *type, int3 *c_map, float *activity, char *spike, unsigned int num, int sub_t){
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
    float da = 0;
    if(id < num){
        float a = activity[id];
        da = -a/TAU;
        a = ( spike[ num*sub_t + id] )? 1.0: a + da;
        activity[id] = a;
        color[ id ] = make_float4( ((float) c_map[type[id]].x)/255.f*a , ((float) c_map[type[id]].y)/255.f*a, ((float) c_map[type[id]].z)/255.f*a, (a > 0.2)?a:0 );
    }
}
/////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){

#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    if(false == initGL(&argc, argv)){
        fprintf(stderr, "ERROR: initGL Error");
        exit(1);
    }


    createVBOs(&vbo, &vbo_resources, 2, NUM_OF_NEURONS * 4 * sizeof(float), cudaGraphicsMapFlagsWriteDiscard);


    glutDisplayFunc(display2);

    glutMainLoop();

    return 0;
}

/////////////////////////////////////////////////////////////////////////

void LoadPos( float4 *pos, char *type_arr, GLfloat frame_vertex[8][3], const char *filename, int num ){
    FILE *fp;
    if( (fp = fopen( filename, "r+")) == NULL ){
        fprintf(stderr, "%s can't open\n", filename);
        exit(1);
    }
    char s[256];
    int c = 0;
    float x_range[2] = {0,0};
    float y_range[2] = {0,0};
    float z_range[2] = {0,0};
    while( fgets(s, 256, fp) != NULL){
        int id, type;
        double x, y, z;
        sscanf(s, "%d %d %lf %lf %lf", &id, &type, &x, &y, &z);
        
        x_range[0] = (x_range[0] > x/100)? x/100: x_range[0];
        x_range[1] = (x_range[1] < x/100)? x/100: x_range[1];
        y_range[0] = (y_range[0] > y/100)? y/100: y_range[0];
        y_range[1] = (y_range[1] < y/100)? y/100: y_range[1];
        z_range[0] = (z_range[0] > z/100)? z/100: z_range[0];
        z_range[1] = (z_range[1] < z/100)? z/100: z_range[1];


        if( id <= num ){
            pos[id-1] = make_float4( x/100, y/100, z/100, 1.0f);
            type_arr[id-1] = type;
        }else{
            fprintf(stderr, "id %d is out of range %d\n", id, num);
        }
        c++;
    }
    fprintf(stderr, "xrange %f - %f\n", x_range[0], x_range[1]);
    fprintf(stderr, "yrange %f - %f\n", y_range[0], y_range[1]);
    fprintf(stderr, "zrange %f - %f\n", z_range[0], z_range[1]);
    

        frame_vertex[ 0 ][0] = x_range[0];
        frame_vertex[ 0 ][1] = y_range[0];
        frame_vertex[ 0 ][2] = z_range[0];

        frame_vertex[ 1 ][0] = x_range[0]; 
        frame_vertex[ 1 ][1] = y_range[0]; 
        frame_vertex[ 1 ][2] = z_range[1]; 

        frame_vertex[ 2 ][0] = x_range[0]; 
        frame_vertex[ 2 ][1] = y_range[1]; 
        frame_vertex[ 2 ][2] = z_range[1]; 

        frame_vertex[ 3 ][0] = x_range[0];
        frame_vertex[ 3 ][1] = y_range[1];
        frame_vertex[ 3 ][2] = z_range[0];

        frame_vertex[ 4+0 ][0] = x_range[1];
        frame_vertex[ 4+0 ][1] = y_range[0];
        frame_vertex[ 4+0 ][2] = z_range[0];

        frame_vertex[ 4+1 ][0] = x_range[1]; 
        frame_vertex[ 4+1 ][1] = y_range[0]; 
        frame_vertex[ 4+1 ][2] = z_range[1]; 

        frame_vertex[ 4+2 ][0] = x_range[1]; 
        frame_vertex[ 4+2 ][1] = y_range[1]; 
        frame_vertex[ 4+2 ][2] = z_range[1]; 

        frame_vertex[ 4+3 ][0] = x_range[1];
        frame_vertex[ 4+3 ][1] = y_range[1];
        frame_vertex[ 4+3 ][2] = z_range[0];

    fprintf(stderr, "%d positions have loaded.\n", c);
    return;
}

void LoadSpike( FILE **fp, char *spike, int num, int width ){
    static int isFirst = true;
    static float old_t = 0.f;
    static int sub_id = 0;
    int count = 0;
    memset( (void*)spike, 0, sizeof(char)*num*width );

    if( !isFirst ){
        spike[sub_id] = 1;
    }

    char s[256];
    char *err_s;
    while( (err_s = fgets(s, 256, *fp) ) != NULL){
        int id;
        double t;

        sscanf(s, "%lf %d %*d", &t, &id);
        if(id >= num){
            fprintf(stderr, "id %d is out of range %d\n ->:%s\n", id, num, err_s);
            continue;
        }
        if( old_t < t ){
            count++;
            old_t = t;
            if( !(count < width) ){
                sub_id = id;
                break;
            }else{
                spike[ num*( ((int)t)%width ) + id] = 1;
            }
        }else{
            spike[ num*( ((int)t)%width ) + id] = 1;
        }
    }

    if(err_s == NULL){
        fclose(*fp);
        *fp = fopen( "spike_data/spike.dat", "r" );
        fprintf(stderr, "file_reset\n");
        isFirst = true;

    }else{
        isFirst = false;
    }

}


/////////////////////////////////////////////////////////////////////////


bool initGL(int *argc, char **argv){
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUDA first sample");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);
    //glDepthFunc(GL_ALWAYS);

    glViewport(0, 0, window_width, window_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA , GL_ONE);

    return true;
}

//////////////////////////////////////////////////////
////! Create VBO
void createVBOs(GLuint **local_vbo_list, struct cudaGraphicsResource ***local_vbo_resources, int vbo_num, unsigned int size, unsigned int vbo_res_flags){
    *local_vbo_list = (GLuint *)malloc(sizeof(GLuint)*vbo_num);
    *local_vbo_resources = (struct cudaGraphicsResource **)malloc( sizeof(struct cudaGraphicsResource *)*vbo_num );


    glGenBuffers(vbo_num, *local_vbo_list );
    for(int i = 0; i < vbo_num; i++){
        glBindBuffer(GL_ARRAY_BUFFER, (*local_vbo_list)[i]);
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        checkCudaErrors( cudaGraphicsGLRegisterBuffer( &(*local_vbo_resources)[i], (*local_vbo_list)[i], vbo_res_flags ) );
    }

    return;
}

////! Delete VBO
void deleteVBOs(GLuint *vbo, struct cudaGraphicsResource **vbo_res, int vbo_num){
    for(int i = 0; i < vbo_num; i++){
        cudaGraphicsUnregisterResource(vbo_res[i]);
        glBindBuffer(1, vbo[i]);
        glDeleteBuffers(1, &vbo[i]);
        vbo[i] = 0;
    }
    return;
}

/////! Draw box line
int edge[12][2] = {
    {0,1},{1,2},{2,3},{3,0},
    {4,5},{5,6},{6,7},{7,4},
    {0,4},{1,5},{2,6},{3,7}
};
GLfloat frame_pos[8][3];
void draw_box_wire( GLfloat pos[8][3] ){

    glLineWidth(0.1f);
    glColor4f(1.0, 1.0, 1.0, 1.0f);
    glBegin(GL_LINES);
    for(int i=0;i<12;i++){
        glVertex3fv( pos[ edge[i][0] ] );
        glVertex3fv( pos[ edge[i][1] ] );
    }
    glEnd();
    glLineWidth(1.f);

}

//////////////////////////////////////////////////////
float g_fAnim = 0.f;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

float fov = 60.0f;
int mouse_buttons = 0;
int mouse_old_x, mouse_old_y;
float cen_dy = 0.f;

float *d_activity;
char *d_spike;
char h_spike[NUM_OF_NEURONS * PRE_LOAD * 2];
FILE *display_fp;
int display_isFirst = true;

char *d_type;
int3 *d_cmap;
int3 h_cmap[] = { {255,51,51}, {255,153,51},{255,255,51},{153,255,51},{51,255,51},{51,255,153},{51,255,255},{51,153,255},{51,51,255},{153,51,255},{255,51,255},{255,51,153},{160,160,160} };

int sub_t = 0;
pthread_t p_th[2];
cudaStream_t stream[2];

const float view_port_radius = 10.f;
float phi = M_PI*3/5;
float theta = 0; //M_PI*3/5;
float middle[3] = {0.f,0.f,0.f};

void display2(){


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, window_width, window_height);
    gluPerspective( fov, (GLfloat)window_width/(GLfloat) window_height, 0.1, 100.0);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    middle[0] = (frame_pos[0][0]+frame_pos[6][0])/2;
    middle[1] = (frame_pos[0][1]+frame_pos[6][1])/2;
    middle[2] = (frame_pos[0][2]+frame_pos[6][2])/2;

    gluLookAt(  middle[0]+view_port_radius*cosf(phi)*cosf(theta) , middle[1]+view_port_radius*sinf(theta), middle[2]+view_port_radius*sinf(phi)*cosf(theta),
                middle[0], middle[1] + cen_dy, middle[2],
                0.,1.,0.);


    glEnable(GL_ALPHA_TEST);
    //run CUDA
    cudaGraphicsMapResources(2, vbo_resources, 0);
    float4 *pos, *col;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer( (void **)&pos, &num_bytes, vbo_resources[0] );
    cudaGraphicsResourceGetMappedPointer( (void **)&col, &num_bytes, vbo_resources[1] );

    if( display_isFirst ){
        cudaMalloc( &d_activity, sizeof(float)*NUM_OF_NEURONS ) ;
        cudaMalloc( &d_spike, sizeof(char)*NUM_OF_NEURONS*PRE_LOAD*2 );
        cudaMalloc( &d_type, sizeof(char)*NUM_OF_NEURONS*PRE_LOAD );
        cudaMalloc( &d_cmap, sizeof(int3)*13 );

        cudaMemcpy( d_cmap, h_cmap, sizeof(int3)*13, cudaMemcpyHostToDevice );

        float4 *h_pos = (float4 *)malloc(sizeof(float4)*NUM_OF_NEURONS);
        char *h_type = (char *)malloc(sizeof(char)*NUM_OF_NEURONS);
        fprintf(stderr, "h_pos: %p\n", h_pos);
        LoadPos( h_pos, h_type, frame_pos, "positions.dat", NUM_OF_NEURONS);

        cudaMemcpy( pos, h_pos, sizeof(float4)*NUM_OF_NEURONS, cudaMemcpyHostToDevice );
        cudaMemcpy( d_type, h_type, sizeof(char)*NUM_OF_NEURONS, cudaMemcpyHostToDevice );

        cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&stream[1], cudaStreamNonBlocking);


        init_act_spike<<< (NUM_OF_NEURONS + 127)/128, 128>>>( d_activity, d_spike, NUM_OF_NEURONS );

        display_fp = fopen( "spike_data/spike.dat", "r");
        if(display_fp == NULL)exit(1);

        LoadSpike( &display_fp, h_spike, NUM_OF_NEURONS, PRE_LOAD);
        cudaMemcpy( &d_spike, &h_spike, sizeof(char)*NUM_OF_NEURONS*PRE_LOAD, cudaMemcpyHostToDevice );
        fprintf(stderr, "init_done\n");


        display_isFirst = false;
        //free(h_pos);
    }


    // load spike
    if( sub_t % PRE_LOAD == 0 && g_fAnim > 0){
        fprintf(stderr, "invoke sync\n");
        int i = !(sub_t/PRE_LOAD);
        cudaStreamSynchronize(stream[i]);
        fprintf(stderr, "synchronized.\n");

        fprintf(stderr, "///////////////////////////// %lf %d %d %d\n", g_fAnim, PRE_LOAD , sub_t/PRE_LOAD, i);
        LoadSpike( &display_fp, &h_spike[i*NUM_OF_NEURONS*PRE_LOAD], NUM_OF_NEURONS, PRE_LOAD);
        cudaMemcpyAsync( &d_spike[ i*NUM_OF_NEURONS*PRE_LOAD ], &h_spike[ i*NUM_OF_NEURONS*PRE_LOAD ], sizeof(char)*NUM_OF_NEURONS*PRE_LOAD, cudaMemcpyHostToDevice, stream[i] );
    }

    update_activity<<< (NUM_OF_NEURONS+127)/128, 128 >>>(col, d_type, d_cmap, d_activity, d_spike, NUM_OF_NEURONS, sub_t);

    cudaGraphicsUnmapResources(2, vbo_resources, 0);
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glColorPointer(4, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glDrawArrays(GL_POINTS, 0, NUM_OF_NEURONS );

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    draw_box_wire( frame_pos );


    glDisable(GL_ALPHA_TEST);
    glDisable(GL_DEPTH_TEST);
    glutSwapBuffers();
    g_fAnim += 0.01f;
    sub_t = (sub_t + 1 < PRE_LOAD*2)? sub_t + 1 : 0;
}

void display(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    gluLookAt(0.0,0.7,0.7, 0.,0.,0.,0.,1.,0.);

    //run CUDA
    cudaGraphicsMapResources(2, vbo_resources, 0);
    float4 *pos, *col;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer( (void **)&pos, &num_bytes, vbo_resources[0] );
    cudaGraphicsResourceGetMappedPointer( (void **)&col, &num_bytes, vbo_resources[1] );


    dim3 block(8, 8, 1);
    dim3 grid(mesh_width/ block.x, mesh_height / block.y, 1);
    simple_vbo_kernel<<< grid, block >>>( pos, col, mesh_width, mesh_height, g_fAnim);
    fprintf(stderr, "\r%lf", g_fAnim );

    cudaGraphicsUnmapResources(2, vbo_resources, 0);
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glColorPointer(4, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);


    glutSwapBuffers();
    g_fAnim += 0.01f;
}


void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void keyboard(unsigned char key, int , int ){
    switch(key){
        case 119: //w
            theta += M_PI/180*D_ANGLE;
            break;
        case 97: // a
            phi += M_PI/180*D_ANGLE;
            break;
        case 115: // s
            theta -= M_PI/180*D_ANGLE;
            break;
        case 100: // d
            phi -= M_PI/180*D_ANGLE;
            break;
        default:
            fprintf(stderr,"key: %d\n", key);
            break;
    }
}
void mouse(int button, int state, int x, int y)
{
    switch(button){
        case 3: // up
            fov -= 1.0f;
            break;
        case 4: // down
            fov += 1.0f;
            break;
    }
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
        cen_dy = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y){
    float dx, dy;
    dy = (float)(y - mouse_old_y);
    if( mouse_buttons & 1){
        cen_dy += dy * 0.01f;
    }else{
        cen_dy = 0;
    }
    mouse_old_x = x;
    mouse_old_y = y;
}

