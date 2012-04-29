#define MAX_URAND 0xFFFFFFFFL
void usrand (unsigned seed);
unsigned urand0 (void);
unsigned urand (void);
double randreal(void);
int randint(int,int);
void initialize_rangen(int);
int randsign();
double randgen(void);
double randgauss(void);
void randgauss2(double,double,double,double *,double *);
