#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

#include <blas.h>
#include <math.h>

#include "blockio.h"
#include "util.h"

#define row 0
#define col 1
#define beg 0
#define end 1


double (*vget)(vector*, const size_t) = &vector_get;
void   (*vset)(vector*, const size_t, double) = &vector_set;
double (*mget)(matrix*, const size_t, const size_t) = &matrix_get;
void   (*mset)(matrix*, const size_t, const size_t, double) = &matrix_set;

typedef vector vect;
typedef matrix matr;

vect* (*vect_alloc)(const size_t) = &vector_alloc;
void  (*vect_free)(vect*) = &vector_free;
matr* (*matr_alloc)(const size_t, const size_t) = &matrix_alloc;
void  (*matr_free)(matr*) = &matrix_free;

typedef struct netw_struct {
     matr** weights;
     size_t size;
} netw;

typedef struct flow_struct {
     vect** layer;
     size_t size;
} flow;

void assert_matr_matr_mul(matr* a, matr* b) {
     assert(a->size2 == b->size1);}
void assert_matr_matr(matr* a, matr* b) {
     assert(a->size1 == b->size1);
     assert(a->size2 == b->size2);
}
void assert_vect_matr(vect* v, matr* m) {
     assert(m->size1 == v->size);}
void assert_matr_vect(matr* m, vect* v) {
     assert(m->size2 == v->size); }
void assert_vect_vect(vect* a, vect* b) {
     assert(a->size == b->size); }
void assert_valid_p(void* p) {
     assert(p != NULL); };

/*
set value, but relative to approach [apr]
*/
void vset_p(vect* v, int index, int apr,  double val) {
     if (apr == end) index = v->size - index - 1;
     vset(v, index, val);
}

/*
set value, but relative to approach [apr1] and [apr2]
*/
void mset_p(matr* m, int index1, int apr1, int index2, int apr2,  double val) {
     if (apr1 == end) index1 = m->size1 - index1 - 1;
     if (apr2 == end) index2 = m->size2 - index2 - 1;
     mset(m, index1, index2, val);
}

/*
get column or row of matrix
*/
vector_view sub_vect(matr* m, int apr, int mode, int index) {
     int max_i;
     vector_view view;
     if (mode == row)  max_i = m->size1;
     else if (mode == col)	max_i = m->size2;
     if (apr == end) index = max_i - index - 1;
     if (mode == row)  view = matrix_row(m, index);
     else if (mode == col)  view = matrix_column(m, index);
     return view;
}


/*
who stole this???
*/
double vector_sum(const vector* a) {
     double sum = 0;
     for (size_t i = 0; i < a->size; i++) {
          sum += vget(a, i);
     }
     return sum;
}

/*
Fill a vector with (kinda random) values
*/
void vect_rnd_fill(vect* v) {
     for (size_t i = 0; i < v->size; i++) {
          vset(v, i, (double)i + 1);
     }
}

/*
Fill a vector with true random values
*/
void vect_trng_fill(vect* v, rng* rng, unsigned long range) {
     for (size_t i = 0; i < v->size; i++) {
          double res = (double) rng_uniform_int(rng, range) - range / 2;
          vset(v, i, res);
     }
}

/*
Fills a vector with gradient from cost vector and signal vector
*/
void vect_cost_fill(vect* grad, vect* sig, vect* cost) {
     assert_vect_vect(grad, cost);
     assert_vect_vect(grad, sig);
     for (size_t i = 0; i < grad->size; i++) {
          vset(grad, i, 2 * (vget(sig, i) - vget(cost, i)));
     }
}


/*
Fill a matrix with (kinda random) values,
such that it becomes a diagonal matrix.
*/
void matrix_rnd_fill(matr* m) {
     for (size_t i = 0; i < m->size1; i++) {
          for (size_t j = 0; j < m->size2; j++) {
               mset(m, i, j, (double)j + i + 1);
          }
     }
}

/*
Creates a new network with slots for [size] layers
note that the layers themselves aren't created
*/
netw* netw_alloc(size_t size) {
     netw* n = malloc(sizeof(netw));
     assert(n != NULL);
     n->size = size;
     n->weights = calloc(size, sizeof(size_t));
     assert(n->weights != NULL);
     return n;
}

/*
Frees the network and all layers in its slots
note that the slots must not be empty (indicative of a bug)
*/
void netw_free(netw* n) {
     for (size_t i = 0; i < n->size; i++) {
          assert(n->weights[i] != NULL);
          matr_free(n->weights[i]);
     }
     free(n->weights);
     free(n);
}

/*
Creates a new network
layers will be created and added to the network slots as well
The layer sizes are determined by [spec]
*/
netw* netw_init(vect* spec) {
     netw* n = netw_alloc(spec->size - 1);
     for (size_t i = 0; i < spec->size - 1; i++) {
          n->weights[i] = matr_alloc((size_t)vget(spec, i + 1), (size_t)vget(spec, i));
     }
     return n;
}


/*
Fills all the layers in a network with (kinda random) value
*/
void netw_rnd_fill(netw* n) {
     for (size_t i = 0; i < n->size; i++) {
          matrix_rnd_fill(n->weights[i]);
     }
}

/*
Fill the network with some number
*/
void netw_set_fill(netw* n, double d) {
     for (size_t i = 0; i < n->size; i++) {
          matrix_set_all(n->weights[i], d);
     }
}


/*
clear the final node, in all except the last layer of weights
*/
void netw_bias_clear(netw* n) {
     for (size_t i = 0; i < n->size - 1; i++) {
          vector_view node = sub_vect(n->weights[i], end, row, 0);
          vector_set_all(&node.vector, 0);
     }
}


/*
set bias weight, in all except the last layer of weights
*/
void netw_bias_set(netw* n) {
     for (size_t i = 0; i < n->size - 1; i++) {
          mset_p(n->weights[i], 0, end, 0, end, 1.0);
     }
}


/*
subtracts one network from another
*/
void netw_sub(netw* a, netw* b) {
     assert(a->size == b->size);
     for (size_t i = 0; i < a->size; i++) {
          assert_matr_matr(a->weights[i], b->weights[i]);
          matrix_sub(a->weights[i], b->weights[i]);
     }
}

/*
adds one network to another
*/
void netw_add(netw* a, netw* b) {
     assert(a->size == b->size);
     for (size_t i = 0; i < a->size; i++) {
          assert_matr_matr(a->weights[i], b->weights[i]);
          matrix_add(a->weights[i], b->weights[i]);
     }
}

/*
multiplies one network by a const
*/
void netw_scale(netw* a, double b) {
     for (size_t i = 0; i < a->size; i++) {
          matrix_scale(a->weights[i], b);
     }
}


/*
Prints the network
[debugging method]
*/
void netw_print(netw* n) {
     for (size_t i = 0; i < n->size; i++) {
          print_matrix(n->weights[i]);
     }
}

/*
Creates a new flow with slots for [size] layers
note that the layers themselves aren't created
*/
flow* flow_alloc(size_t size) {
     flow* f = malloc(sizeof(flow));
     assert(f != NULL);
     f->size = size;
     f->layer = malloc(sizeof(size_t) * size);
     assert(f->layer != NULL);
     return f;
}

/*
Frees the network and all layers in its slots
note that the slots must not be empty (indicative of a bug)
*/
void flow_free(flow* f) {
     for (size_t i = 0; i < f->size; i++) {
          assert(f->layer[i] != NULL);
          vect_free(f->layer[i]);
     }
     free(f->layer);
     free(f);
}

/*
Creates a new flow
layers will be created and added to the flow slots as well
The layer sizes are determined by [spec]
note: flow size will always be one more of network size
*/
flow* flow_init(vect* spec) {
     flow* f = flow_alloc(spec->size);
     for (size_t i = 0; i < f->size; i++) {
          f->layer[i] = vect_alloc((size_t)vget(spec, (size_t)i));
     }
     return f;
}

/*
Prints the flow
[debugging method]
*/
void flow_print(flow* n) {
     for (size_t i = 0; i < n->size; i++) {
          print_vector(n->layer[i]);
     }
}

/*
Creates a new vector from a C array
[utility method]
WARNING: ADDS 1 TO ALL EXCEPT LAS
*/
vect* quick_spec(size_t nums[], size_t size) {
     vector* spec = vect_alloc(size);
     for (size_t i = 0; i < size; i++) {
          vset(spec, i, (double)nums[i] + (i != size - 1));
     }
     return spec;
}

/*
Creates a new vector from a C array
[utility method]
*/
vect* quick_vect(size_t nums[], size_t size) {
     vector* spec = vect_alloc(size);
     for (size_t i = 0; i < size; i++) {
          vset(spec, i, (double)nums[i]);
     }
     return spec;
}

/*
Pass the signal[prv] forward through the layer[layer] into[nxt] while setting the relu input activation [grad]
CRITICAL FUNCTION FOR PERFORMANCE
*/
void layer_feval(vect* prv, vect* nxt, vect* grad, matr* layer) {
     blas_dgemv(CblasNoTrans, 1, layer, prv, 0, nxt);
     for (size_t i = 0; i < nxt->size; i++) {
          double act = nxt->data[i];
          nxt->data[i] = act > 0 ? act : 0;
          grad->data[i] = act > 0 ? 1 : 0;
     }
}

/*
Propagate the signal through the network while setting relu input activation grad
*/
void netw_feval(netw* netw, flow* sig, flow* grad) {
     for (size_t i = 0; i < netw->size; i++) {
          layer_feval(sig->layer[i], sig->layer[i + 1], grad->layer[i], netw->weights[i]);
     }
}

/*
Pass the layer [prv] backwards through the transposed weigths [layer] into nxt [nxt] while applying input gradients [out]
*/
void layer_beval(vect* prv, vect* nxt, vect* grd, matr* layer) {
     assert_matr_vect(layer, nxt);
     assert_vect_matr(prv, layer);
     assert_vect_vect(nxt, grd);
     blas_dgemv(CblasTrans, 1, layer, prv, 0, nxt);
     vector_mul(nxt, grd);
}

/*
Backpropagate aux through the network while using relu input activation grad
*/
void netw_beval(netw* netw, flow* b_pass, flow* relu_act) {
     for (int i = (int)netw->size - 1; i > 0; i--) {
          layer_beval(b_pass->layer[i], b_pass->layer[i - 1], relu_act->layer[i - 1], netw->weights[i]);
     }
}

/*
Apply the auxilary partial product [aux_b] from the b pass, and the signals, to compute a gradient for the weights
*/
void layer_grad(matr* grad, vect* aux_b, vect* sig) {
     assert(grad->size2 == sig->size);
     assert(grad->size1 == aux_b->size);
     matrix_view sig_row = matrix_view_vector(sig, 1, sig->size);
     matrix_view aux_col = matrix_view_vector(aux_b, aux_b->size, 1);
     blas_dgemm(CblasNoTrans, CblasNoTrans, 1, &aux_col.matrix, &sig_row.matrix, 1, grad);
}

/*
Compute the gradient
*/
void netw_grad(netw* grad, flow* b_pass, flow* sig) {
     for (int i = (int)grad->size - 1; i >= 0; i--) {
          layer_grad(grad->weights[i], b_pass->layer[i], sig->layer[i]);
     }
}

/*
  TODO: future optimization to save aux from the heap
*/
void netw_grad_ex_cheap(netw* netw_t, netw* grad, flow* relu_act, flow* sig, vect* cost) {
}

/*
cost function
sum of the squares of the differences
*/
double cost_f(vect* sig, vect* target) {
     assert(sig->size == target->size);
     double sum = 0;
     for (size_t i = 0; i < sig->size; i++) {
          sum += pow_2(vget(sig, i) - vget(target, i));
     }
     return sum;
}

/*
Contains all the memory that the model requires
also contains some common parameters
*/
typedef struct framework_struct {
     netw* net;
     flow* f_pass;
     flow* aux_b;
     flow* relu_act;
     vect* spec;
     size_t minor_last;
     size_t major_last;
     size_t dim_last;
     size_t dim_first;
} framework;

/*
Initialize the memory using the network and spec. The spec and net must match.
*/
framework* fm_init(netw* net, vect* spec) {
     framework* fm = malloc(sizeof(framework));
     assert(fm != NULL);
     fm->spec = spec;
     fm->major_last = spec->size - 1;
     fm->minor_last = spec->size - 2;
     fm->dim_last = (size_t)vget(spec, spec->size - 1);
     fm->dim_first = (size_t)vget(spec, 0);
     fm->net = net;                                    // The network
     vector_view spec_nfirst = vector_subvector(spec, 1, fm->major_last);
     fm->relu_act = flow_init(&spec_nfirst.vector);    // Weighed inputs of each layer
     fm->f_pass = flow_init(spec);                     // Signal outputs of each layer
     fm->aux_b = flow_init(&spec_nfirst.vector);       // aux partial product for backprop
     return fm;
}

/*
Free the memory
*/
void fm_free(framework* fm) {
     flow_free(fm->relu_act);
     flow_free(fm->f_pass);
     flow_free(fm->aux_b);
     free(fm);
}

/*
Eval the network [fm] at the point [input] and store in [output]
*/
void fm_eval(framework* fm, vect* input, vect* output) {
     assert_matr_vect(fm->net->weights[0], input);                 // Assert input matches first weights
     assert_vect_matr(output, fm->net->weights[fm->minor_last]);   // Assert ouput matches last weights
     vector_memcpy(fm->f_pass->layer[0], input);               // Copy input, into head of flow
     netw_feval(fm->net, fm->f_pass, fm->relu_act);                // Eval network (relu_act not used)
     vector_memcpy(output, fm->f_pass->layer[fm->major_last]); // Copy tail of flow, into output
}

/*
Compute the gradient at point [fm] and store in [tot_grad]. pairs (in, out) are kept constant. The gradient seeks to minimize the distance to [out]
*/
void fm_der(framework* fm, vect* in, vect* out, netw* tot_grd) {
     assert_matr_vect(fm->net->weights[0], in);                                                  // Assert input matches first weights
     assert_vect_matr(out, fm->net->weights[fm->minor_last]);                                    // Assert output matches last weights
     vector_memcpy(fm->f_pass->layer[0], in);                                                // Copy input, into head of forward pass
     netw_feval(fm->net, fm->f_pass, fm->relu_act);                                              // Eval network
     vect_cost_fill(fm->aux_b->layer[fm->minor_last], fm->f_pass->layer[fm->major_last], out);   // Fill tail of backwards pass, with cost
     vector_mul(fm->aux_b->layer[fm->minor_last], fm->relu_act->layer[fm->minor_last]);      // Init tail of backwards pass, with relu_act
     netw_beval(fm->net, fm->aux_b, fm->relu_act);                                               // Backpass
     netw_grad(tot_grd, fm->aux_b, fm->f_pass);                                                  // Fill gradient
}
