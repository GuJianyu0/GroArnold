//#include "inc.h"
extern "C"{
    #include "inc.h"
}

int main(){

    clock_t start,end;
    start = clock();

    double x,y,z, pot;
    get_parameter_(); // read coefficients from file 'scficoef',
                     // model settings & parameters from file 'scfmod', 'scfpar'
   
    long int N_total = 1e6/1e2; //N_potcal=N_total*500*20*10
    long int N_other = 1e6/1e6; //1e6/1e2;
    long int N_each = 10/10;
    long int N_shut = 10; //parallel and so on

    double x0 = -100., y0 = -100., z0 = -1.;
    double dx = x0/N_total;
    long int N_print = 0;
   
    for(long int ix=0;ix<N_total;ix++){
        for(long int iy=0;iy<N_other;iy++){
            for(long int iz=0;iz<N_each;iz++){
            x = x0+dx*ix;
            y = y0+dx*iy;
            z = z0+dx*iz;
            get_pot_(&x,&y,&z,&pot);
            N_print++;
            if(N_print%1000==0)
                printf("potential = %.3e; %ld\n", pot, N_print);
            }
        }
    }

    end = clock();
    // printf("%f, %f\n", (double)end, (double)start);
    printf("time = %f\n",(double)(end-start)/1e6);
    //1e8 times, ?(~1e4) particles: 375.5 secs.



    f20211008(3.1415926);

    printf("Write done.\n");
    return 0;
}