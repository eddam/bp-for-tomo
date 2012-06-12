#include <stdlib.h>
#include <math.h>

double atanh_th_th(double x,double y){  /* returns  argth(th(x)*th(y)) */
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

