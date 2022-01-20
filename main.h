#include <stdarg.h>

#include <linalg.h>

#include "core.h"
#include "str_builder.h"

/*
A datapoint. Contains (x, y) and a buffer for the (y) the network actually gives
Merely used for caching, real datapoints are stored in a matrix, or created dynamically using rng
*/
typedef struct point_struct {
     vect* xval;
     vect* ynet;
     vect* ytar;
} point;


/*
a dataset for training
*/
typedef struct dataset_struct {
     int pos;
     int max;
} dataset;


/*
Create a datapoint
*/
point* point_init(framework* spec) {
     point* state = malloc(sizeof(point));
     assert(state != NULL);
     state->xval = vect_alloc(spec->dim_first);
     state->ynet = vect_alloc(spec->dim_last);
     state->ytar = vect_alloc(spec->dim_last);
     return state;
}

/*
Free a datapoint
*/
void point_free(point* state) {
     vect_free(state->xval);
     vect_free(state->ynet);
     vect_free(state->ytar);
     free(state);
}


/*
100 data points from -50 to 50
the function to fit is f(x) = x + max(0, x - 50)
*/
int td_m_next(dataset* data, point* pt) {
     assert(pt->xval->size == 2);
     assert(pt->ytar->size == 1);
     double x = (double)data->pos;
     double on = x > 50;
     double y = x + on * x - on * 50;

     vset(pt->xval, 0, x);
     // vset(pt->xval, 1, data->pos);
     vset_p(pt->xval, 0, end, 1.0);
     vset(pt->ytar, 0, y);
     data->pos++;
     if (data->pos == data->max) {
          data->pos = 0;
          return 0;
     } else {
          return 1;
     }
}


/*
Compute the gradient at point [pos] and store in [vel]. Pairs (in, out) are given by the iterator [points] and [next]
Will consume points until max_steps is reached, or no points are left. 
*/
double next_gradient(framework* pos, netw* vel, point* pt, void* points, int next(void*, point*)) {
     unsigned max_steps = UINT_MAX;
     netw_set_fill(vel, 0);                              // reset gradient
     bool has_more = 1;                                  // keep going?
     double error = 0;                                   // least squares error
     unsigned i = 0; 
     while (has_more && i < max_steps) {
          i++;                          
          has_more = next(points, pt);                   // load next point
          fm_der(pos, pt->xval, pt->ytar, vel);          // compute the gradient      
          vector_memcpy(pt->ynet,                    // 
               pos->f_pass->layer[pos->major_last]);     // Internal hack - gets evaluated val without calling
          error += cost_f(pt->ytar, pt->ynet);           // add to least squares
     }
     double scale_rate = 1 / (double)i;                  // scale error and gradients
     netw_bias_clear(vel);                               // don't modify these
     netw_scale(vel, scale_rate);
     return error * scale_rate;
}

/*
Compute the error at point [pos] and store in [vel]. Pairs (in, out) are given by the iterator [points] and [next]
Will consume points until max_steps is reached, or no points are left.
*/
double next_error(framework* pos, point* pt, void* points, int next(void*, point*)) {
     unsigned max_steps = UINT_MAX;
     bool has_more = 1;                                  // keep going?
     double error = 0;                                   // least squares error
     unsigned i = 0;
     while (has_more && i < max_steps) {
          i++;
          has_more = next(points, pt);                   // load next point
          fm_eval(pos, pt->xval, pt->ynet);              // compute the value      
          error += cost_f(pt->ytar, pt->ynet);           // add to least squares
     }       
     return error * (1 / (double)i);                     // scale error
}


#define scale(mult) netw_scale(vel, mult); step_size *= mult;
/*
Starts by scaling the step_size down a lot, then increase it exponentially until the error starts decreasing
returns this step_size
*/
double minimize_diagonal(framework* pos, netw* vel, point* pt, void* points, int next(void*, point*)) {
     double step_size = 1;                                  // keeps track of the scale
     double prev_error = INFINITY;                          // as bad as possible
     double error = next_error(pos, pt, points, next);      
     double og_error = error;                               // for debugging
     scale(-0.000001);                                       // something tiny idk
     while (error < prev_error) {  
          prev_error = error;
          scale(2);                                         // double the step
          netw_add(pos->net, vel);                          // walk in the direction
          error = next_error(pos, pt, points, next);        // try it out
          netw_sub(pos->net, vel);                          // walk back
     }
     // if (og_error == prev_error) exit(-4);
     scale(0.5);                                            // compensate for overshooting
     netw_scale(vel, 1 / step_size);                        // vel shouldn't be scaled here, but in the function that called this
     return step_size;
}


/*
TODO: future optimization so that the gradient isn't scaled down by a random value
*/
double minimize_diagonal_fast(framework* pos, netw* vel, point* pt, void* points, int next(void*, point*)) { return 0; }

/*
Some vision so we aren't totally blind
*/
void print_step(netw* pos, netw* vel, double score, double rate) {
     printf("RATE: %.2e\nSCORE: %.2e\n", rate, score);
     puts("----------------POSITION-----------------");
     netw_print(pos);
     puts("----------------VELOCITY-----------------");
     netw_print(vel);
     printf("\n\n\n");
}

/*
detect anything that might go wrong in the gradient descent, then shut down
some errors are only checked against [error] as its assumed that [prev_error] was once called as [error]
*/
bool is_error(double prev_error, double error) {
     if (prev_error < error) {
         // puts("Uh-oh, you're going the wrong direction"); return true;
     }
     if (prev_error == error) {
         // puts("Oh noooo, you're stuck :("); return true;
     }
     if (isinf(error)) {
          puts("WOAH, way too fast"); return true;
     }
     if (!isfinite(error)) {
          puts("something is wrong"); return true;
     }
     return false;
}

/*
Trains the network [pos] on the dataset [points/next] using [steps] with adaptive stride
*/
void adaptive_learn(framework* pos, int steps, void* points, int next(void*, point*)) {
     netw* vel = netw_init(pos->spec);    // The gradient
     point* point = point_init(pos);      // A point which gradient finder caches to 
     double error = INFINITY;             // error starts at infity: i.e. as bad as possible
     int diag_hz = steps / 10;            // frequency of outputs
     print_vbar("DESCENDING");
     for (int i = 0; i < steps; i++) {
          double prev_error = error;                                       // save the error, before computing next
          error = next_gradient(pos, vel, point, points, next);            // sets vel
          double rate = minimize_diagonal(pos, vel, point, points, next);  // computes optimal learning_rate/step_size by mimizing in the direction of gradient
          netw_scale(vel, rate);                                           // scales the gradient by optimal
          if (i % diag_hz  == 0) print_step(pos->net, vel, error, rate);   // print stats
          if (is_error(prev_error, error)) {                               // gradient descent is wild
               if (i % diag_hz != 0) print_step(pos->net, vel, error, rate);// in case step was skipped, print the final step
               exit(2);
          };                       
          netw_add(pos->net, vel);                                         // move
     }
     netw_free(vel);
     point_free(point);
}


/*
Evaluate the network against the data set, then save results to files
*/
void test_net(framework* net, dataset* data, int next(dataset*, point*)) {
     point* p = point_init(net);
     data->pos = 0;
     matr* xvals = matr_alloc(data->max, p->xval->size - 1);
     matr* yvals = matr_alloc(data->max, p->ynet->size);
     matr* targets = matr_alloc(data->max, p->ytar->size);
     int i = 0;
     bool has_more = 1;
     while (has_more) {
          has_more = next(data, p);
          vector_view xrow = sub_vect(xvals, beg, row, i);
          vector_view yrow = sub_vect(yvals, beg, row, i);
          vector_view tar = sub_vect(targets, beg, row, i);
          fm_eval(net, p->xval, &yrow.vector);
          vector_view subvect = vector_subvector(p->xval, 0, p->xval->size - 1);
          vector_memcpy(&xrow.vector, &subvect.vector);
          vector_memcpy(&tar.vector, p->ytar);
          i++;
     }
     save_txt_matrix(xvals, "net-x.txt");
     save_txt_matrix(yvals, "net-y.txt");
     save_txt_matrix(targets, "tar-y.txt");
     point_free(p);
}




/*
Quickly evaluates network [fm] using the params in double var args. Returns nothing, but prints result
*/
void quick_eval(framework* fm, ...) {
     str_builder_t* sb = str_builder_create();
     str_builder_add(sb, "f(");
     va_list args;
     va_start(args, fm);
     vect* x = vect_alloc(fm->dim_first);
     vect* y = vect_alloc(fm->dim_last);
     for (size_t i = 0; i < x->size; i++) {
          double next_val = va_arg(args, double);
          str_builder_add_double(sb, next_val);
          str_builder_add_char(sb, ',');
          vset(x, i, next_val);
     }
     va_end(args);
     str_builder_add(sb, ") = \n");
     print_vbar("");
     str_builder_print(sb);
     fm_eval(fm, x, y);
     print_vector(y);
     str_builder_destroy(sb);
     vect_free(x);
     vect_free(y);
}

int main(int argc, char* argv[]) {
     size_t sp[] = { 1, 2, 1 }; vect* spec = quick_spec(sp, sizeof(sp) / sizeof(size_t));
     netw* net = netw_init(spec);                                           // Allocate network 
     netw* de = netw_init(spec);                                            // Allocate derivative 
     netw_set_fill(net, 0.1);                                               // fill network with zeroes
     netw_fill_random(net);
     netw_bias_clear(net);                                                  // init dummy
     netw_bias_set(net);
     framework* fm = fm_init(net, spec);                                    // init framework

     print_vbar("fitting f(x,y)=x+y+100 in the range [-5, 5]");
     dataset* points = malloc(sizeof(dataset));
     assert(points != NULL);
     points->max = 100;
     points->pos = 0;
     test_net(fm, points, td_m_next);

     adaptive_learn(fm, 100000, points, td_m_next);
     quick_eval(fm, 0.0, 1.0);
     test_net(fm, points, td_m_next);

     free(points);
     fm_free(fm);
     netw_free(net);
     netw_free(de);
     vect_free(spec);
     printf("");
}

