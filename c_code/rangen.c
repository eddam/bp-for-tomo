#include <math.h>
void initialize_rangen(int seed){
	usrand(seed);
}
double randreal()
{
//	return (rand()/(RAND_MAX+1.0));
	return (urand()/((double)MAX_URAND+1.0));	//rajoute le +1 ici et en dessous le 08-01-2007
}


int randint(int from,int upto)
{
	int ijk;
	upto=upto-from+1;
	ijk=(int)(((double)upto*(double)urand())/((double)MAX_URAND+1.0));
	if(ijk==upto) {
		ijk=(int)(((double)upto*(double)urand())/((double)MAX_URAND+1.0));
		if(ijk==upto) ijk=upto-1;
	}
	return(from+ijk);
}


/*
	urand(), urand0() return uniformly distributed unsigned random ints
		all available bits are random, e.g. 32 bits on many platforms
	usrand(seed) initializes the generator
		a seed of 0 uses the current time as seed
	
	urand0() is the additive number generator "Program A" on p.27 of Knuth v.2
	urand() is urand0() randomized by shuffling, "Algorithm B" on p.32 of Knuth v.2
	urand0() is one of the fastest known generators that passes randomness tests
	urand() is somewhat slower, and is presumably better
*/

static unsigned rand_x[56], rand_y[256], rand_z;
static int rand_j, rand_k;

void usrand (unsigned seed)
{
	int j;
	rand_x[1] = 1;
	if (seed) rand_x[2] = seed;
	else rand_x[2] = time (NULL);
	for (j=3; j<56; ++j) rand_x[j] = rand_x[j-1] + rand_x[j-2];
		rand_j = 24;
	rand_k = 55;
	for (j=255; j>=0; --j) urand0 ();
	for (j=255; j>=0; --j) rand_y[j] = urand0 ();
	rand_z = urand0 ();
}
unsigned urand0 (void)
{
	if (--rand_j == 0) rand_j = 55;
	if (--rand_k == 0) rand_k = 55;
	return rand_x[rand_k] += rand_x[rand_j];
}
unsigned urand (void)
{
	int j;
		j =  rand_z >> 24;
	rand_z = rand_y[j];
	if (--rand_j == 0) rand_j = 55;
	if (--rand_k == 0) rand_k = 55;
	rand_y[j] = rand_x[rand_k] += rand_x[rand_j];
	return rand_z;
}
int randsign(){
	 int j;
	 j=randint(0,1);
	 if(j==0){
		 return(-1);}
	 else {
		 return(1);}
}
/* Random gaussian varaible, centered( <x>=0) and of width 1: (<x*x>=1).   */
double randgauss(){
	double x,y,r,phi;
	x=randreal();
	y=randreal();
	if(x==0) x=randreal();
	r=sqrt(-2.*log(x));
	phi=2.*3.141592654*y;
	return r*cos(phi);
}
double randexp(double a){ //distribution aexp(-a*x)
	double x;
	x=randreal();
	if(x==0) x=randreal();
	return -log(x)/a;
}
void randperm(int n,int *a){ //a[1],...a[n] is a random permutation of 1,...,n
	int i,j,ainew;
	for (i=1;i<=n;i++) a[i]=i;
	for (i=1;i<=n-1;i++){
		j=randint(i,n);
		ainew=a[j];
		a[j]=a[i];
		a[i]=ainew;
	}
}

/* randgauss2 generates a pair of random varaibles, x,y, distributed
according to the joint gaussian distribution Log P(x,y)= -(1/2)(a*x*x+b*y*y +2*c*x*y).
It first generates x from its marginal, and then generates y from its conditional probability given x. Both
are gaussian and generated from randgauss().*/
		
void randgauss2(double a, double b,double c,double *x,double *y){
  if(a*b-c*c <=0){
    printf("randgauss2 impossible, a=%f b=%f c=%f\n",a,b,c);
    exit(2);
  }
  (*x)=randgauss()*pow(a-c*c/b,-.5);
  (*y)=-(*x)*c/b+randgauss()*pow(b,-.5);
}
