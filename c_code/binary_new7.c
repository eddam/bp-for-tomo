
/*
Binary tomography with BP  and a ferromagnetic prior.
Processing of images of size N= L*L
microc_threshold decides on when we switch to micronanonical decoding of 1d system.

compile with 
gcc binary_new7.c -O2


Assez facile converge en 14 iterations, 131 secondes: 
time ./a.out 256 50  12  1  100 100 100 .9  10 123 
avec damping=.5, converge en 2 iterations, 38 secondes
avec damping=.0, converge en 4 iterations, 73 secondes

difficile avec plein de petites ellipses, avec damping=.5, converge en 4 iterations, 125 secondes:
time ./a.out 256 150  24  1  100 100 100 0.5  10 123

Echantillon de S. Roux avec 256*256 pixels: 
résolu en 5 iterations, 6 minutes par: time ./a.out 0  150  24  1  100 100 100 0.5  10 123
avec threshold=20 au lieu de 100, la resolution prend 10 iterations, 76 secondes par:
time ./a.out 0 150  24  1  100 100 20 0.5  10 123 
avec threshold=20 et damping =.2 au lieu de .5, la resolution prend 9 iterations, 61 secondes

Echantillon de 400*400 avec 300 petites ellipses:
./a.out 400 300  30  1  100 100 20 0.5  10 123 -> résolu en 6 iterations, 2 min 40s.
time ./a.out 400 200  30  1  100 100 20 0  10 123  -> résolu en 5 iterations, 2 min 20s.

 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "rangen.h"
#include "rangen.c"
#include "nrutil.h"
#include "nrutil.c"
#include "sort2_MM.c"

#define PI 3.14159
#define SQR2 1.41421

int verbose=2;
double BIGFIELD;

/*******************************************************************
                 Geometry
**************************************************/

/******** indexsite orders the sites of a two dimensional square lattice of length L 
i is the index of the line, going from 1 to L
j is the index of the column, going from 1 to L
indexsite goes from 1 to L**2**/
int indexsite(int L,int i,int j){
  return 1+L*(i-1)+(j-1);
}
//Given the index of a site, finds the line and column in the matrix.
void matrixcoordinates(int L,int indsite,int *i,int * j){
  int ii,jj;
  ii=(indsite-1)/L;
  (*i)=ii+1;
  jj=indsite-1-L*ii;
  (*j)=jj+1;
}
/* Geometry: for each site of the square matrix, finds the coordinates of the center of the pixel cp=(cpx,cpy)
 */
void pixel_pos(int L, double *cpx,double *cpy){
  int n,i,j;
  for(n=1;n<=L*L;n++){
    matrixcoordinates(L,n,&i,&j);
    cpx[n]=i-.5;
    cpy[n]=L-(j-.5);
    if(verbose>8) printf("test pixel_pos: n=%i i=%i j=%i  %f %f  \n",n,i,j,cpx[n],cpy[n]);
  }
}
// Generate an image, ellipse or superposition of ellipses
void genimage(int L, int numell,int **ima,double *cpx,double *cpy){
  int i,j,n,iell;
  double *x0,*y0,*a,*b;//parameters of the ellipses
  x0=dvector(1,numell);  y0=dvector(1,numell);  a=dvector(1,numell);  b=dvector(1,numell);
  for(iell=1;iell<=numell;iell++){
    x0[iell]=L/2+ .9*(L/2)*(2.*randreal()-1);
    y0[iell]=L/2+ .9*(L/2)*(2.*randreal()-1);
      a[iell]=(1+10.*randreal());
      b[iell]=(1+10.*randreal());
    
    /*a[iell]=(1+1.*randreal());   for test on very small images
      b[iell]=(1+1.*randreal());
    */
  }
  for(n=1;n<=L*L;n++){
    matrixcoordinates(L,n,&i,&j);
    ima[i][j]=-1;
  }
  for(n=1;n<=L*L;n++){
    matrixcoordinates(L,n,&i,&j);
    for(iell=1;iell<=numell;iell++){
      if (pow(cpx[n]-x0[iell],2)/pow(a[iell],2)+pow(cpy[n]-y0[iell],2)/pow(b[iell],2)<1) ima[i][j]=1;
    }
  }
}


/*******************************************************************
                  Measurements: Radon transform
********************************************************************/

double tmax(int L,double theta){
  if(theta<0) {
    printf("theta<0\n");
    exit(2);
  }
  if(theta<PI/2.) return L*(cos(theta)+sin(theta));
  else return L*sin(theta);
}
double tmin(int L,double theta){
  if(theta<0) {
    printf("theta<0\n");
    exit(2);
  }
  if(theta<PI/2.) return 0;
  else return L*cos(theta);
}

void rotate_image(int N,double *cpx,double *cpy,double theta){
  int n;
  double cc,ss,cxt,cyt;
  cc=cos(theta);
  ss=sin(theta);
  for(n=1;n<=N;n++){
    cxt=cpx[n];
    cyt=cpy[n];
    cpx[n]=cxt*cc-cyt*ss;
    cpy[n]=cxt*ss+cyt*cc;
  }
}
void calc_ybounds(int N,double *cpyrot,double *ymin,double *ymax){
  int n;
  *ymin=N;
  *ymax=-N;
  for(n=1;n<=N;n++){
    if(cpyrot[n]>*ymax) *ymax=cpyrot[n];
    if(cpyrot[n]<*ymin) *ymin=cpyrot[n];
    //printf("test n=%i y=%f ymin=%f ymax=%f \n",n,cpyrot[n],*ymin,*ymax);
  }
  *ymax=*ymax+.5;
  *ymin=*ymin-.5;
}
double calcjeff(double jcoup,double x1,double x2,double y1,double y2){
  double dist,res;
  dist=fabs(x1-x2)+fabs(y1-y2);// Manhattan distance
  res=pow(tanh(jcoup),dist);
  res=.5*log1p(res)-.5*log1p(-res);
  if(verbose>10){
    printf("calcjeff: %f %f %f %f %f ; dist=%f tanh**dist=%f res=%f\n",jcoup,x1,x2,y1,y2,dist,pow(tanh(jcoup),dist),res);
  }
  return res;
}
void calc_measurements(int L,int Lmax, int numdir, int **ima, int ** varnum,double *yy,int *dir,int *size,int **vois,int *numvois,double **jj,double jcoup, int *Mtrue){
  int mu,sum,i,j,k;
  int N=L*L,n,bin[L*L+1],sizebin[2*L+1],indpix[2*L+1][2*L+1],ibin,ibinmax,indspin[2*L+1];
  double cpx[L*L+1],cpy[L*L+1],theta,ymin,ymax,xx[2*L+1];
  mu=0;
  for (k=1;k<=L*L;k++) numvois[k]=0;

  //
  for(theta=0;theta<PI-.00001;theta+=PI/numdir){
    pixel_pos(L,cpx,cpy);
    rotate_image(N,cpx,cpy,theta);
    calc_ybounds(N,cpy,&ymin,&ymax);
    printf("theta=%f ymin=%f ymax=%f\n",theta,ymin,ymax);
    for(ibin=0;ibin<=2*L;ibin++) sizebin[ibin]=0;
    for(n=1;n<=N;n++) {
      bin[n]=cpy[n]-ymin+1;// bin belongs to 1,...,(int)(ymax-ymin)+1;
      if(bin[n]>2*L) {printf("cata bin\n");exit(2);}
      sizebin[bin[n]]++;
      indpix[bin[n]][sizebin[bin[n]]]=n;
    }
    // For each bin, numbered as ibin, we have the number of spins in it, sizebin[ibin] and their list indpix[ibin,1]... indpix[ibin,sizebin[ibin]
    ibinmax= 1+ymax-ymin;
    if(verbose>8){
      for(ibin=1;ibin<=ibinmax;ibin++) {
	printf("bin number %i size %i list:",ibin,sizebin[ibin]);
	for(i=1;i<=sizebin[ibin];i++) printf(" %i ",indpix[ibin][i]);
	printf("\n");
      }
    }
    for(ibin=1;ibin<=ibinmax;ibin++) {
      if(sizebin[ibin]>0){
	mu++;
	size[mu]=sizebin[ibin];
	// build ordered list of the spins in the constraint mu, ordered by the value of their x coordinate
	for(i=1;i<=sizebin[ibin];i++) {
	  indspin[i]=indpix[ibin][i];
	  xx[i]=cpx[indpix[ibin][i]];
	}
	sort2_MM(sizebin[ibin],xx,indspin);
	if(verbose>15){
	  printf("bin number %i size %i ordered list:",ibin,sizebin[ibin]);
	  for(i=1;i<=sizebin[ibin];i++) printf(" %i %f ;",indspin[i],xx[i]);
	  printf("\n");
	}
	for(i=1;i<=size[mu];i++) varnum[mu][i]=indspin[i];// i-ieme voisin dans mu
	for(i=1;i<size[mu];i++) {
	  if(verbose>10) printf("...  measurement mu=%i. Compute the coupling between variables %i and %i\n",mu,varnum[mu][i],varnum[mu][i+1]);
	  jj[mu][i]=calcjeff(jcoup,cpx[varnum[mu][i]],cpx[varnum[mu][i+1]],cpy[varnum[mu][i]],cpy[varnum[mu][i+1]]);
	}
	sum=0;
	for(n=1;n<=size[mu];n++) {
	  k=varnum[mu][n];
	  numvois[k]++;
	  vois[k][numvois[k]]=mu;
	  //printf("TESTTEST mu=%i k=%i numvois=%i vois=%i\n",mu,k,numvois[k],vois[k][numvois[k]]);
	  matrixcoordinates(L,k,&i,&j);
	  sum+= ima[i][j];
	}
	yy[mu]=sum;
	dir[mu]=theta;
      }
    }
  }
  *Mtrue=mu;
  for(mu=1;mu<=*Mtrue;mu++){
    if(size[mu]>Lmax){
      printf("DESASTRE mu=%i size=%i Lmax=%i\n",mu,size[mu],Lmax);
    }
  }
  if(verbose>4){
    for(mu=1;mu<=*Mtrue;mu++){
      printf("Measurement mu=%i, size=%i, list=",mu,size[mu]);
      for(n=1;n<=size[mu];n++){
	printf("%i ",varnum[mu][n]);
	matrixcoordinates(L,varnum[mu][n],&i,&j);
	printf("ima(%i,%i)=%i ;",i,j,ima[i][j]);
      }
      printf("\n   couplings=");
      for(i=1;i<size[mu];i++) printf("%f ",jj[mu][i]);
      printf("   yy=%f\n",yy[mu]);
    }
    for(i=1;i<=L*L;i++){
      printf("  variable %i belongs to these %i checks :",i,numvois[i]);
      for(n=1;n<=numvois[i];n++) printf(" %i",vois[i][n]);
      printf("\n");
    }
  }
}

//*************************************************************************
/**************************************************************************
  Solution of a one dimensional Ising chain, microcanonical and canonical
****************************************************************************/

double atanh_th_th(double x,double y){	/* returns  argth(th(x)*th(y)) */
  double yy,xx,xm,ym,res;
  int isigne;
  if(x*y>=0)  isigne=1;
  else isigne=-1;
  xx=fabs(x);
  yy=fabs(y);
  if(xx<yy) {
    xm=xx;
    ym=yy;
  }
  else {
    xm=yy;
    ym=xx;
  }
  //res= xm-.5*log((1+exp(-2*(ym-xm)))/(1+exp(-2*(xx+yy))));
  res=xm-.5*log1p(exp(-2*(ym-xm)))+.5*log1p(exp(-2*(xx+yy)));
  return ((double)isigne)*res;
}

/* Canonical: solve-chain solves a chain of nn spins, with nearest couplings jj, such that the sum of the spins
is yy (implemented in canonical ensemble: constraint on the average), with local fields hh. It computes all the local magnetizations mag[i], and returns for each site hloc[i]=atanh(mag[i])
 */
double mag_chain(int nn,double *hh,double *uu,double *vv,double *mag,double *jj,double *hloc,double hext,double epsilon){
  int i;
  double magtot; /* uu are the messages to the right, vv those to the left.
			  hext is the external magnetic field adapted to impose the
			  global constraint on the magnetization
			*/
  uu[1]=0;
  for(i=2;i<=nn;i++) uu[i]=atanh_th_th(jj[i-1],hext+hh[i-1]+uu[i-1]);
  vv[nn]=0;
  for(i=nn-1;i>=1;i--) vv[i]=atanh_th_th(jj[i-1],hext+hh[i+1]+vv[i+1]);
  magtot=0;
  for(i=1;i<=nn;i++){
    hloc[i]=uu[i]+vv[i]+hh[i]+hext;
    if(hloc[i]>BIGFIELD)hloc[i]=BIGFIELD;
    if(hloc[i]<-BIGFIELD)hloc[i]=-BIGFIELD;
    mag[i]=tanh(hloc[i]);
    magtot+=mag[i];
  }
  return magtot;
}
void solve_chain(int nn,double *hh,double *jj,double y,double *hloc,double *hhext){
  int iter;
  double *uu,*vv,*mag,magtot,hext,epsilon=.0001; /* uu are the messages to the right, vv those to the left.
			  hext is the external magnetic field adapted to impose the
			  global constraint on the magnetization
			*/
  double hmax,hmin;
  uu=dvector(1,nn); vv=dvector(1,nn); mag=dvector(1,nn);
  if(fabs(y)>nn){printf("DESASTRE\n");exit(2);}
   hext=0;
  magtot=mag_chain(nn,hh,uu,vv,mag,jj,hloc,hext,epsilon);
  if(magtot<y){
    hmin=0;
    hext=8.;
    while(y-mag_chain(nn,hh,uu,vv,mag,jj,hloc,hext,epsilon)>epsilon){
      hmin=hext;
      hext=hext*2.;
      if(hext>100000000) {printf("hext>100000000\n"); exit(2);}
    }
    hmax=hext;
  }
  else {
    hmax=0;
    hext=-8.;
    while(mag_chain(nn,hh,uu,vv,mag,jj,hloc,hext,epsilon)-y>epsilon){
      hmax=hext;
      hext=hext*2.;
    if(hext<-10000000000) {printf("hext<-10000000\n"); exit(2);}
    }
    hmin=hext;
  }
  magtot=2.*nn;
  
  iter=0;
  //hmax=4;
  //hmin=-4;
   while(fabs(magtot-y)>epsilon){
  //while(hmax-hmin>epsilon){
    iter++;
    hext=.5*(hmax+hmin);
    magtot=mag_chain(nn,hh,uu,vv,mag,jj,hloc,hext,epsilon);
    if(magtot<y)hmin=hext;
    else hmax=hext;
    if(iter>200) {
      printf("Unsolved after %i iterations, %i %f %f %f\n",iter, nn, y, magtot,hext);
      return;
    }
  }
  if(verbose>3){
    printf("Chain nn=%i solved after niter=%i hext=%f magtot=%f y=%f mag and hloc:\n",nn,iter,hext,magtot,y);
  }
  *hhext=hext;
  free_dvector(uu,1,nn);  free_dvector(vv,1,nn);  free_dvector(mag,1,nn);
  return;
}
//Protected version of  log(exp(a)+exp(b))
double logexpplusexp(double a,double b,double MININF){
  double x,y;
  if(a==MININF){
    if(b==MININF) return MININF;
    else return b;
  }
  if(b==MININF) return a;
  if(a>b) {x=a; y=b;}
  else {x=b;y=a;}
  return x+log1p(exp(y-x));
}

/*** Full microcanonical solution ********/
void solve_chain_microc(int nn,double *hh,double *jj,double y,double *hloc,double *hhext){// with exact constraint on the global magnetization = uy
  int i,jkl,ss,uu,inty;
  double MININF=-10000;
  double **ar,**al,**br,**bl,*mag,fp,fm;
  double **testmat;
  testmat=dmatrix(1,10,-5,5);
  free_dmatrix(testmat,1,10,-5,5);
  ar=dmatrix(1,nn,-nn-1,nn+1); br=dmatrix(1,nn,-nn-1,nn+1);/*right moving matrices.
					   ar[p][j] is the free entropy of the chain to the left of spin p
					   when the last spin is + and the mag is j
					   br[p][j] is the free entropy when the last spin is - and the mag is j 
					   The second argument goes from -n-1 to n+1, but the value of the message at -n-1, or the one at n+1, 
					   are always -infinity (i e in practice MININF)*/
  al=dmatrix(1,nn,-nn-1,nn+1); bl=dmatrix(1,nn,-nn-1,nn+1);// left moving
  mag=dvector(1,nn);
 
  inty=(int)y;
  if(fabs(y)>nn){printf("DESASTRE\n");exit(2);}
  if((abs(nn-y))%2){printf("DESASTRE PARITE\n");exit(2);}
  for(jkl=-nn-1;jkl<=nn+1;jkl++) {ar[1][jkl]=MININF;br[1][jkl]=MININF;al[nn][jkl]=MININF;bl[nn][jkl]=MININF;}
  // Compute right-moving messages
  ar[1][1]=hh[1];
  br[1][-1]=-hh[1];
  i=0;
  if(verbose>=15){
      printf("colonne %i , right moving messages: \n",i+1);
      for(jkl=nn;jkl>=-nn;jkl--) printf(" %i %f %f \n",jkl, ar[i+1][jkl],br[i+1][jkl]);
  }
  for(i=1;i<=nn-1;i++){
    for(jkl=-i-1;jkl<=i+1;jkl+=2){
      ar[i+1][jkl]=logexpplusexp(ar[i][jkl-1]+jj[i],br[i][jkl-1]-jj[i],MININF)+hh[i+1];
      br[i+1][jkl]=logexpplusexp(ar[i][jkl+1]-jj[i],br[i][jkl+1]+jj[i],MININF)-hh[i+1];
    }
    for(jkl=-i-1-2;jkl>=-nn-1;jkl-=2){
      ar[i+1][jkl]=MININF;
      br[i+1][jkl]=MININF;
    }
    for(jkl=i+1+2;jkl<=nn+1;jkl+=2){
      ar[i+1][jkl]=MININF;
      br[i+1][jkl]=MININF;
    }
    if(verbose>=15){
      printf("colonne %i , right moving messages: \n",i+1);
      for(jkl=nn;jkl>=-nn;jkl--) printf(" %i %f %f \n",jkl, ar[i+1][jkl],br[i+1][jkl]);
      /*jkl=2;
	printf("TEST jkl=2: %f %f %f\n",ar[i][jkl+1]-jj[i],br[i][jkl+1]+jj[i],logexpplusexp(ar[i][jkl+1]-jj[i],br[i][jkl+1]+jj[i],MININF));*/
    }
  }
  // Compute left-moving messages
  al[nn][1]=hh[nn];
  bl[nn][-1]=-hh[nn];
  i=nn-1;
   if(verbose>=15){
      printf("------\n colonne %i , left moving messages: \n",i+1);
      for(jkl=nn;jkl>=-nn;jkl--) printf(" %i %f %f \n",jkl, al[i+1][jkl],bl[i+1][jkl]);
    }
  for(i=1;i<=nn-1;i++){
    for(jkl=-i-1;jkl<=i+1;jkl+=2){
      al[nn-i][jkl]=logexpplusexp(al[nn-i+1][jkl-1]+jj[nn-i],bl[nn-i+1][jkl-1]-jj[nn-i],MININF)+hh[nn-i];
      bl[nn-i][jkl]=logexpplusexp(al[nn-i+1][jkl+1]-jj[nn-i],bl[nn-i+1][jkl+1]+jj[nn-i],MININF)-hh[nn-i];
    }
    for(jkl=-i-1-2;jkl>=-nn-1;jkl-=2){
      al[nn-i][jkl]=MININF;
      bl[nn-i][jkl]=MININF;
    }
    for(jkl=i+1+2;jkl<=nn+1;jkl+=2){
      al[nn-i][jkl]=MININF;
      bl[nn-i][jkl]=MININF;
    }
    if(verbose>=15){
      printf("colonne %i , left moving messages: \n",nn-i);
      for(jkl=nn;jkl>=-nn;jkl--) printf(" %i %f %f \n",jkl, al[nn-i][jkl],bl[nn-i][jkl]);
    }
  }
  // Compute the local magnetizations, conditioned on the total magnetization
  //printf("*********************\n Remember y=%f inty=%i nn=%i\n",y,inty,nn);
  for(i=1;i<=nn;i++){
    ss=1;
    uu=i;
    if(fabs(y-uu+ss)<=nn) fp=ar[i][uu]+al[i][inty-uu+ss]-hh[i]*ss;//subtract hh[i]*ss since it is already counted twice in ar and al
    else fp=MININF;

    for(uu=i+2;uu<=nn;uu+=2){
      if(fabs(y-uu+ss)<=nn){
	fp=logexpplusexp(fp,ar[i][uu]+al[i][inty-uu+ss]-hh[i]*ss,MININF);
      }
    }
    for(uu=i-2;uu>=-nn;uu-=2){
      if(fabs(y-uu+ss)<=nn){
	fp=logexpplusexp(fp,ar[i][uu]+al[i][inty-uu+ss]-hh[i]*ss,MININF);
      }
    }    
    ss=-1;
    uu=i;
    if(fabs(y-uu+ss)<=nn) fm=br[i][uu]+bl[i][inty-uu+ss]-hh[i]*ss;//subtract hh[i]*ss since it is already counted twice in br and bl
    else fm=MININF;
    for(uu=i+2;uu<=nn;uu+=2){
      if(fabs(y-uu+ss)<=nn){
	fm=logexpplusexp(fm,br[i][uu]+bl[i][inty-uu+ss]-hh[i]*ss,MININF);
      }
    }
    fflush(stdout);
    for(uu=i-2;uu>=-nn;uu-=2){
      if(abs(inty-uu+ss)<=nn){
	fm=logexpplusexp(fm,br[i][uu]+bl[i][inty-uu+ss]-hh[i]*ss,MININF);
	fflush(stdout);
      }
    }
    hloc[i]=.5*(fp-fm);
    if(hloc[i]>BIGFIELD)hloc[i]=BIGFIELD;
    if(hloc[i]<-BIGFIELD)hloc[i]=-BIGFIELD;
  }
  free_dmatrix(ar,1,nn,-nn-1,nn+1); 
  free_dmatrix(br,1,nn,-nn-1,nn+1); 
  free_dmatrix(al,1,nn,-nn-1,nn+1); 
  free_dmatrix(bl,1,nn,-nn-1,nn+1); 
  free_dvector(mag,1,nn);
  return ;
}

/*******************************************************************
                   Message passing
********************************************************************/
void  initfield(int M,double **field,int *size,double *yy){
  int mu,r;
  double x;
  for(mu=1;mu<=M;mu++){
    x=yy[mu]/size[mu];
    if(fabs(x)<1){
      x=.5*(log1p(x)-log1p(-x));
    }
    else{
      if(x>0) x=BIGFIELD;
      else x=-BIGFIELD;
    }
    if(x>BIGFIELD/2) x=BIGFIELD/2;
    if(x<-BIGFIELD/2) x=-BIGFIELD/2;
    for(r=1;r<=size[mu];r++) field[mu][r]=x;
    //for(r=1;r<=size[mu];r++) .1*(randreal()-.5);
  }
}
/* calc_field: updates all messages h_{mu to i}, returns the result in hloc  */
void calc_field(int mu,int *size, double microc_threshold,double onsager,int **varnum,int **varnuminv,double *yy,double **jj,  double **hatf,double *hloc,int *typecalc){
  double *hh,*jchain,hext,magtot;
  int nlocked,k,i,r;
  hh=dvector(1,size[mu]);
  jchain=dvector(1,size[mu]);
  nlocked=0;
  magtot=0;
  hext=0;
  for(k=1;k<=size[mu];k++) {
    i=varnum[mu][k];
    r=varnuminv[mu][k];
    hh[k]=hatf[i][r];
    if(hh[k]>=BIGFIELD) {
      hh[k]=BIGFIELD;
      nlocked++;
      magtot++;
    }
    if(hh[k]<=-BIGFIELD){
      hh[k]=-BIGFIELD;
      nlocked++;
      magtot--;
    }
  }
  //if(nlocked>0) printf("%i %i %i %i ;",mu,size[mu],nlocked,(int)magtot);
  if(verbose>14)for(k=1;k<=size[mu];k++) printf("%f ",hh[k]); 
  if(verbose>14) printf("\n");
  if((nlocked==size[mu])&&(fabs(magtot-yy[mu]))<.01){
    for(k=1;k<=size[mu];k++) hloc[k]=1.5*hh[k];//CHanged in version 15: factor 3
    if((int)fabs(yy[mu])!=size[mu])printf(" non-trivial frozen constraint:  mu=%i, size=%i nlocked=%i magtot=%f yy=%f\n",mu,size[mu],nlocked,magtot,yy[mu]);
    *typecalc=0;
  }
  else{
    for(k=1;k<size[mu];k++) jchain[k]=jj[mu][k];
    if(size[mu]-nlocked>microc_threshold){
      solve_chain(size[mu],hh,jchain,yy[mu],hloc,&hext);//Solve the chain number mu
      *typecalc=2;
    }
    else {
      solve_chain_microc(size[mu],hh,jchain,yy[mu],hloc,&hext);//Solve the chain number mu 
      *typecalc=1;
    }
  }
  for(k=1;k<=size[mu];k++) hloc[k]=hloc[k]-onsager*hh[k];
  free_dvector(hh,1,size[mu]);
  free_dvector(jchain,1,size[mu]);
}



void calc_hatf(int N,int L,double **field,double **hatf,double *htot,int *numvois,int **vois,int **voisinv,int **varnum,double *frustr,double *hftyp, int **ima, double *mse){
  int mu,i,s,k,numtyp,x,y;
  double sum,frust,frustav;
  frustav=0;
  numtyp=0;
  *hftyp=0;
  *mse=0;
  for(i=1;i<=N;i++){
    sum=0;
    frust=0;
    for(k=1;k<=numvois[i];k++) {
      mu=vois[i][k];
      s=voisinv[i][k];
      //next line is just a check. Can delete it (and take out varnum) later
      if(varnum[mu][s]!=i) {printf("Pb voisinv \n");exit(2);}
      sum += field[mu][s];
      frust+= fabs(field[mu][s]);
    }
    if(sum>0) s=1;
    else s=-1;
    matrixcoordinates(L,i,&x,&y);
    if(s!=ima[x][y]) (*mse)=(*mse)+1;
    frust=(frust-fabs(sum))/frust;
    frustav+= frust;
    //sum is the total local field \hat h_i. In order to compute \hat h_{i \to \mu} one needs to substract the part coming from mu
    htot[i]=sum;
    for(k=1;k<=numvois[i];k++){
      mu=vois[i][k];
      hatf[i][k]=sum-field[mu][voisinv[i][k]];// champ chapeau
      (*hftyp) +=fabs(hatf[i][k]);
      numtyp++;
    }
  }
  *frustr=frustav/N;
  (*hftyp)=(*hftyp)/numtyp;
  (*mse)=(*mse)/N;
}
/* After a change of the field coming from constraint mu to i(i.e. the field[mu][k] where varnum[mu][k]=i),
   one updates here all the hat fields going from i to nu, with nu different from mu*/
void update_hatf(int mu,int k,int **varnum,int **varnuminv,int **vois,int *numvois, double **hatf,double *htot,double fieldchange){
  int i,s,r;
  i=varnum[mu][k];
  htot[i]+=fieldchange;
  s=varnuminv[mu][k];
  if(vois[i][s]!=mu){printf("vois[i][s]!=mu \n");exit(15);}//test (when taken awy, no need for vois in this subroutine
  for(r=1;r<=numvois[i];r++){
    if(r!=s) hatf[i][r]+=fieldchange;
  }
}

void calc_voisinv(int N,int numdir,int ** vois,int *numvois,int *size,int **voisinv,int **varnum){
  int i,mu,r,s,error;
  for(i=1;i<=N;i++){
    for(r=1;r<=numvois[i];r++){
      mu=vois[i][r];
      for(s=1;s<=size[mu];s++){
	if(varnum[mu][s]==i){
	  voisinv[i][r]=s;
	  break;
	}
      }
    }
  }
  //check
  error=0;
  for(i=1;i<=N;i++){
    for(r=1;r<=numvois[i];r++){
      if(varnum[vois[i][r]][voisinv[i][r]] != i) {
	error++;
	if(verbose>10) printf("CHECK i=%i r=%i vois=%i voisinv=%i var=%i \n",i,r,vois[i][r],voisinv[i][r],varnum[vois[i][r]][voisinv[i][r]]);
      }
    }
  }
  if(error>0) {
    printf("error in computation of voisinv, error=%i\n",error);
    exit(2);
  }
}
void calc_varnuminv(int M,int numdir,int ** vois,int *numvois,int *size,int **varnum,int **varnuminv){
  int i,mu,r,s,error;
  for(mu=1;mu<=M;mu++){
    for(r=1;r<=size[mu];r++){
      i=varnum[mu][r];
      for(s=1;s<=numvois[i];s++){
	if(vois[i][s]==mu){
	  varnuminv[mu][r]=s;
	  break;
	}
      }
    }
  }
  //check
  error=0;
  for(mu=1;mu<=M;mu++){
    for(r=1;r<=size[mu];r++){
      if(vois[varnum[mu][r]][varnuminv[mu][r]] != mu) error++;
    }
  }
  if(error>0) {
    printf("error in computation of varnuminv\n");
    exit(2);
  }
}

/**********************************************
                main
*********************************/

int main(int argc, char **argv) {
  int **ima,**varnum,**varnuminv,*dir,*size,*numvois,**vois,**voisinv,*mag_clipped,generate_image,numell,nflip,wrongpolarized,numpolarized,error;
  double *yy,*cpx,*cpy,**field,**hatf,*hloc,jcoup,**jj,*jchain,err_mess,onsager,frustr,hftyp,mse,sum,damping,fieldav,fieldnew;
  int L,Lmax,N,M,Mmax,seed,n,i,j,k,mu,numdir,iter,niter,sizemax,sizetot,s,writeperiod,numcan,numfrozen,nummicroc,microc_threshold,typecalc,nvmax,wrongspins;
  FILE *ImageOrig,*ImageRec,*Imtoimport,*binary_out;
  double *htot;
  if(argc!=11) {printf("Usage : %s L(0 to read in trybin_256.txt, -1 for the 1025-size one) numell numdir jcoup niter writeperiod microc_threshold damping BIGFIELD  seed \n",argv[0]);exit(1);}
  L=atoi(argv[1]);// L=0 means that one reads the file trybin_256.txt. Any other L means one generates the image, of size L*L
  numell=atoi(argv[2]);// number of ellipses in the image
  numdir=atoi(argv[3]);//number of directions of measurement
  jcoup=atof(argv[4]);
  niter=atoi(argv[5]);// maximal number of iterations
  writeperiod=atoi(argv[6]);//period at which one fprints the spins and the local fields
  microc_threshold=atoi(argv[7]);// in principle, should be 1. But a smaller number, like .5, improves convergence...
  damping=atof(argv[8]);
  BIGFIELD=atof(argv[9]);
  seed=atoi(argv[10]);// seed for random number generator
  onsager=1;
  
  binary_out=fopen("binary_out.dat","w");
  generate_image=1;
  if(L==0){
     L=256;
    generate_image=0;
    Imtoimport=fopen("trybin_256.txt","r");
  }
 if(L==-1){
    L=1025;
    generate_image=0;
    Imtoimport=fopen("trybin.txt","r");
  }
  N=L*L;
  Lmax=2*L;// maximum number of variables involved in one measurement
  printf("L=%i N=%i \n",L,N);
  initialize_rangen(seed);
  ima=imatrix(1,L,1,L);
  Mmax=numdir*L*SQR2;// upper bound on the number of measurements
  varnum=imatrix(1,Mmax,1,Lmax);//varnum[mu][r] gives the index of the r-th spin in the line mu.
  varnuminv=imatrix(1,Mmax,1,Lmax);// s=varnuminv[mu][k] <=> vois[varnum[mu][k]][s]=mu (mu is the s-th neighbour of his k-th neighbour)
  yy=dvector(1,Mmax);//results of measurements
  dir=ivector(1,Mmax);// in what direction was a measurement
  size=ivector(1,Mmax);//number of variables involved in a measurement
  jj=dmatrix(1,Mmax,1,Lmax);//Ising couplings in the chains
  jchain=dvector(1,Lmax);//Ising couplings in one given chain
  numvois=ivector(1,N);// number of measurements in which a varaible appears
  vois=imatrix(1,N,1,numdir);// list of measurements where a variable appears: vois[i][r] is the index of the r-th line where the vertex number i is present
  voisinv=imatrix(1,N,1,numdir);// s=voisinv[i][r] <=> varnum[vois[i][r]][s]=i (i is the s-th neighbour of his r-th neighbour).
  mag_clipped=ivector(1,N);// magnetization found by clipping htot
  htot=dvector(1,N);// full local field found by BP
  // Generate or read the image
  ImageOrig=fopen("ImageOrig.dat","w");
  if(generate_image==1){
    cpx=dvector(1,N);
    cpy=dvector(1,N);
    pixel_pos(L,cpx,cpy);
    genimage(L,numell,ima,cpx,cpy);
    free_dvector(cpx,1,N);
    free_dvector(cpy,1,N);
  }
  else{
   for (i=1;i<=L;i++) {
      for(j=1;j<=L;j++) {
	fscanf(Imtoimport,"%i ",&k);
	if(k==0)ima[i][j]=1;
	else ima[i][j]=-1;
      }
    }
  }
  for(n=1;n<=N;n++){
    matrixcoordinates(L,n,&i,&j);
    fprintf(ImageOrig,"%i ",ima[i][j]);
    //fprintf(ImageOrig,"\n");
  }
  fclose(ImageOrig);
  // Perform the measurements (Radon)
  calc_measurements(L,Lmax,numdir,ima,varnum,yy,dir,size,vois,numvois,jj,jcoup,&M);
  if(M>Mmax){
    printf("M>Mmax, please increase Mmax: %i %i\n",M,Mmax);
    exit(2);
  }
  sum=0;
  for (i=1;i<=L;i++) {
    for(j=1;j<=L;j++) sum+=ima[i][j];
  }
  printf("generated an image with %i pixels and magnetization %i\n",L*L,(int)sum);
  // Geometry of the factor graph and checks
  calc_voisinv(N,numdir,vois,numvois,size,voisinv,varnum);
  calc_varnuminv(M,numdir,vois,numvois,size,varnum,varnuminv);
    sizemax=0;sizetot=0;
  for(mu=1;mu<=M;mu++){
    sizetot+=size[mu];
    if(size[mu]>sizemax) sizemax=size[mu];
    if(verbose>5){
      printf("Constraint mu=%i yy[mu]=%f size=%i:  ",mu,yy[mu],size[mu]);
      for(i=1;i<=size[mu];i++) printf("%i ",varnum[mu][i]);
      printf("\n");
    }
  }
  if(sizemax>Lmax){
    printf("STOP sizemax>Lmax\n");
    exit(12);
  }
  nvmax=0;
  for(i=1;i<=N;i++){
    if(numvois[i]>nvmax)nvmax=numvois[i];
  }
  //Initialisation of the messages
  field=dmatrix(1,M,1,sizemax);// field[mu][r] is the local magnetic field h_{mu to i} where i is the r-th neighbour of mu
  hatf=dmatrix(1,N,1,nvmax);//hatf[i][r] is the local magnetic field \hat h_{i to mu} where mu is the r-th neighbour of i
  initfield(M,field,size,yy);
  calc_hatf(N,L,field,hatf,htot,numvois,vois,voisinv,varnum,&frustr,&hftyp,ima,&mse);
  if(verbose>12){
    printf("Initial field: \n");
    for(mu=1;mu<=M;mu++){
      for(n=1;n<=size[mu];n++) printf("%f ",field[mu][n]);
      printf("\n\n");
    }
    printf("Initial hatf: \n");
    for(n=1;n<=N;n++) {
      for (i=1;i<=numvois[n];i++) printf("%f ",hatf[n][i]);
      printf("\n\n");
    }
  }
  hloc=dvector(1,sizemax);
  printf("%i Measurements done; starting iteration, niter=%i\n",M,niter);fflush(stdout);
  fprintf(binary_out,"# N=%i numdir=%i M=%i BIGFIELD= %f microc_threshold=%i\n",N, numdir, M, BIGFIELD,microc_threshold);
  for(n=1;n<=N;n++) mag_clipped[n]=1;
  //START BP ITERATION HERE!
  for(iter=0;iter<=niter;iter++){
    err_mess=0;
    sizetot=0;
    fieldav=0;
    numcan=0;nummicroc=0;numfrozen=0;
    for(mu=1;mu<=M;mu++){
      //update all messages h_{mu to i}
      if(verbose>10) printf("\n iter=%i: Solving chain mu=%i, with %i spins ",iter,mu,size[mu]);
      calc_field(mu,size,microc_threshold,onsager,varnum,varnuminv,yy,jj,hatf,hloc,&typecalc);
      if(typecalc==0) numfrozen++;
      if(typecalc==1) nummicroc++;
      if(typecalc==2) numcan++;
      for(k=1;k<=size[mu];k++) { // compute field
	err_mess+=fabs(tanh(hloc[k])-tanh(field[mu][k]));
	sizetot++;
	fieldnew=damping*field[mu][k]+(1.-damping)*hloc[k];
	update_hatf(mu,k,varnum,varnuminv,vois,numvois,hatf,htot,fieldnew-field[mu][k]);// update hatf for all constraints nu neighbours of varaible i, except nu;
	field[mu][k]=fieldnew;
	fieldav+=fabs(field[mu][k]);
      }
    }
    //printf("\n");
    err_mess/=sizetot;
    fieldav/=sizetot;
 
    printf("iter= %i,   err_mess=%f, fieldav=%f  ;    frustration=%f, typical hatfield=%f, mse=%e, numcan=%i, nummicroc=%i,numfrozen=%i \n",
	   iter,err_mess,fieldav,frustr,hftyp,mse,numcan,nummicroc,numfrozen);
    fflush(stdout);
    // Monitoring of the progress of BP: compute present image obtained by clipping, and how well it reconstructs
    nflip=0;
    wrongpolarized=0;numpolarized=0;wrongspins=0;
    for(n=1;n<=N;n++){
      if(fabs(htot[n])>2*BIGFIELD) numpolarized++;
      if(htot[n]*mag_clipped[n]<0)nflip++;
      if(htot[n]>0) mag_clipped[n]=1;
      else mag_clipped[n]=-1;
      matrixcoordinates(L,n,&i,&j);
      if(mag_clipped[n]!=ima[i][j]) {
	wrongspins++;
	if(fabs(htot[n])>2*BIGFIELD) wrongpolarized++;
      }
    }
    error=0;
    for(mu=1;mu<=M;mu++){//check the constraints
      sum=0;
      for(k=1;k<=size[mu];k++)	sum+=mag_clipped[varnum[mu][k]];
      if(sum!=yy[mu]) error++;
    }
    printf("Number of wrong spins=%i, fraction wrong = %f ;  nflip=%i ;  number of unsat constraints=%i numpolarized=%i wrongpolarized=%i\n",
	   wrongspins,((double)wrongspins)/N,nflip,error,numpolarized,wrongpolarized);
    if(wrongpolarized>0) printf("ATTENTION wrongpolarized!!!!!!!!!!\n");
    fprintf(binary_out," %i %i %f %i %i\n",iter,(int)mse, mse/N,nflip,error); 
    if(err_mess<.0001) break;//BP has converged
    if(error==0)break;// All constraints are satisfied
  }
  // fprintf the final image obtained by BP reconstruction
  ImageRec=fopen("ImageRec.dat","w");
  for(i=1;i<=L;i++){
    for(j=1;j<=L;j++){
      fprintf(ImageRec,"%i ",mag_clipped[indexsite(L,i,j)]);
    }
    fprintf(ImageRec,"\n");
  }
  fprintf(ImageRec,"\n");
  return 1;
}


