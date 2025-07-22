#ifndef DPD_MDE_H
#define DPD_MDE_H



#define K 13
#define MEMORY_DEPTH 7
#define POPULATION_SIZE 10
#define F_SCALE 0.5f
#define CR_PROB 0.7f
#define MAX_GENERATIONS 200

typedef float data_t;
typedef float phi_t;
typedef float acc_t;
typedef float coef_t;

typedef struct {
    coef_t real;
    coef_t imag;
} ccoef_t;

// DPD function prototype (MDE, memory polynomial)
void dpd_mde(
    data_t i_in[MEMORY_DEPTH], data_t q_in[MEMORY_DEPTH],
    data_t i_ref, data_t q_ref,
    ccoef_t w[K][MEMORY_DEPTH],
    data_t* z_i, data_t* z_q
);

#endif // DPD_MDE_H
