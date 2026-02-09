C***********************************************************************
C
C
C                             File : SCF-code.f
C                                    based on mpiscf.f
C                             Time : 2013-11-12
C
C Original introduction:
C***********************************************************************
C
C
C    A code to evolve self-gravitating systems using a self-consistent
C    field approach.  This version has been optimized for scalar workstations
C    and parallel computers.  The code is written in standard FORTRAN.
C
C    The computational system of units is determined by the input data.
C    No explicit assumtions about the value of the gravitational
C    constant have been made; it is read in as a parameter.
C    Particles are not required to have identical masses.
C
C
C                       Version 4: December1, 1995
C
C
C                    Steinn Sigurdsson: IoA, Cambridge
C                       From original by Lars Hernquist, UCSC
C                    
C    Apr 15 1997: option to multistep the SCF algorithm upgraded
C
C                 Chris Mihos, Johns Hopkins University
C
C  Jan 15, 1999 : various optimizations and conversion to MPI
C                  Bob Leary, San Diego Supercomputer Center
C
C
C=======================================================================
C
C
C     This is the high-level evolution subroutine ascf.  Its tasks are:
C
C          1) to input parameters and the initial system state;
C          2) to perform a diagnostic analysis of the system at
C             each time step (energy, angular momentum, etc.);
C          3) initialise a-dot on first call
C          4) to provide an array of accelerations and a-dots
C           for a master hermite integrator.
C	   5) to advance the SCF particles 
C
C
C
C=======================================================================
C
C
C     Basic global variables/parameters:
C
C          ax,ay,az    : accelerations of bodies.
C          clm, dlm,   : radial functions used to evaluate expansions.
C          elm, flm
C          cputime     : cpu time (secs) used during the simulation.
C          cputime0    : cumulative cpu time at start of run.
C          cputime1    : cumulative cpu time at end of run.
C          dtime       : the timestep.
C          fixacc      : option to force conservation of linear
C                        momentum by setting acceleration of c.o.m.=0.
C          G           : the gravitational constant.
C          headline    : identification string for the run.
C          inptcoef    : option to read in expansion coefficients.
C          lmax        : number of angular eigenfunctions.
C          mass        : masses of bodies.
C          nbodies     : total number of bodies.
C          nbodsmax    : maximum number of bodies.
C          nmax        : number of radial eigenfunctions.
C          noutbod     : frequency of system record outputs.
C          noutlog     : frequency of outputs to log file.
C          nsteps      : number of time-steps to integrate the system.
C          one         : the constant 1.
C          onesixth    : the constant 1/6.
C          outpcoef    : option to write out expansion coefficients.
C          pi          : the constant pi.
C          pot         : potentials of bodies (self-gravity).
C          potext      : potentials of bodies (external field).
C          selfgrav    : option to turn off (.FALSE.) system self-
C                        gravity.
C          tnow        : current system time.
C          tpos        : current position time.
C          tvel        : current velocity time.
C          twoopi      : the constant 2./pi.
C          vx,vy,vz    : velocities of bodies.
C          x,y,z       : positions of bodies.
C          zeroeven    : option to zero out all even terms in the
C                        basis function expansions.
C          zeroodd     : option to zero out all odd terms in the
C                        basis function expansions.
C
C
C-----------------------------------------------------------------------
C
C   Definitions specific to input/output.
C
C          uterm, upars, ulog, ubodsin,   : logical i/o unit numbers.
C            ubodsout,utermfil,uoutcoef,
C            uincoef,ubodsel
C          parsfile, logfile, ibodfile,   : character names of files.
C            obodfile,termfile,outcfile,
C            incfile,elfile
C
C-----------------------------------------------------------------------
C
C   Definitions specific to timing MPI code
C
C          t0mpi,t1mpi,t2mpi,tmpi - real*8 timing variables
C=======================================================================

C***********************************************************************
C
C
        SUBROUTINE calc_a
C        SUBROUTINE accp_LH(nstep)
C
C
C***********************************************************************
C
C
C     Subroutine to compute sinsum and cossum.
C
C
C=======================================================================

        INCLUDE 'tmhscf.h'   
C        INCLUDE 'mpif.h'
        INTEGER k,l,m,n,lmin,lskip,nstep
        LOGICAL firstc
        REAL*8 anltilde,knl,sinth,sinmphi,cosmphi,phinltil,deltam0,
     &         gammln,arggam,sinsum,cossum,coeflm,factrl,
     &         dblfact,ttemp5,ar,ath,aphi,temp2,temp3,temp4,
     &         temp5,temp6,plm,dplm,ultrasp,ultrasp1,ultraspt,clm,
     &         dlm,elm,flm,xi,costh,phi,r,twoalpha,c1,c2,c3,un,unm1,
     &         plm1m,plm2m,rhonltil,alm,blm,cosmphi1,sinmphi1
      real*8 lmm,lplusm,ratio
      real*8 arctanth, cos2theta_m1, sin2thetaminv, tinynum !gjy add: for nan when "costh-1==0"

c note tmpsum must be large enough to hold sinsum1,cossum1,sinsum2,cossum2
c each array is of the form sinsum1(0:nmax,0:lmax,0:lmax)
c with element sinsum1(n,l,m).  But for a given l, m ranges from 0:l.

ccompaq     tmpsum replaced with tmpsum_snd and tmpsum_rcv so that distinct
ccompaq     buffers are used in MPI_Allreduce. Change courtesy of Compaq

        REAL*8 tmpsum_snd((nmax+1)*(lmax+1)*(lmax+2)),
     &         tmpsum_rcv((nmax+1)*(lmax+1)*(lmax+2))
        DIMENSION ultrasp(0:nmax,0:lmax),
     &            ultraspt(0:nmax,0:lmax),ultrasp1(0:nmax,0:lmax),
     &            anltilde(0:nmax,0:lmax),dblfact(lmax+1),
     &            coeflm(0:lmax,0:lmax),sinsum(0:nmax,0:lmax,0:lmax),
     &            cossum(0:nmax,0:lmax,0:lmax),
     &            twoalpha(0:lmax),c1(1:nmax,0:lmax),c2(1:nmax,0:lmax),
     &            c3(1:nmax),cosmphi(0:lmax),sinmphi(0:lmax),
     &            plm(0:lmax,0:lmax),dplm(0:lmax,0:lmax),
     &            knl(0:nmax,0:lmax),twolp1(0:lmax),twolm1(0:lmax),
     &            lplusm(0:lmax,0:lmax),lmm(0:lmax,0:lmax)
        DATA firstc/.TRUE./

        SAVE firstc,dblfact,anltilde,coeflm,lmin,
     &       lskip,twoalpha,c1,c2,c3,knl,twolp1,twolm1,
     &       lmm,lplusm
        
        COMMON/sincos/sinsum,cossum


crss    New declarations (RSS)

        real*8 rxy2,rxyinv,arxyinv(nbodsper) 
        real*8 rxyz2,rxyzinv,arxyzinv(nbodsper)
        
        real*8 mbh, rxyzinv3

C=======================================================================

        IF(firstc) THEN

           firstc=.FALSE.

C         WRITE(*,*) 'Calc coeff...'
           dblfact(1)=1.

           DO 5 l=2,lmax
              dblfact(l)=dblfact(l-1)*(2.*l-1.)
 5         CONTINUE

           DO 20 n=0,nmax
              DO 10 l=0,lmax
                 knl(n,l)=0.5*n*(n+4.*l+3.)+(l+1.)*(2.*l+1.)
                 anltilde(n,l)=-2.**(8.*l+6.)*FACTRL(n)*(n+2.*l+1.5)
                 arggam=2.*l+1.5
                 anltilde(n,l)=anltilde(n,l)*(EXP(GAMMLN(arggam)))**2
           anltilde(n,l)=anltilde(n,l)/(4.*pi*knl(n,l)*FACTRL(n+4*l+2))
 10           CONTINUE
 20        CONTINUE

           DO 25 l=0,lmax

              twoalpha(l)=2.0*(2.*l+1.5)
              twolp1(l)=2.*l+1.
              twolm1(l)=2.*l-1.

              DO 23 m=0,l
                 deltam0=2.
                 IF(m.EQ.0) deltam0=1.
                 coeflm(l,m)=(2.*l+1.)*deltam0*FACTRL(l-m)/FACTRL(l+m)
                 lmm(l,m)=l-m
      if(l.ne.m) lmm(l,m)=1./lmm(l,m)
                 lplusm(l,m)=l+m-1.
 23           CONTINUE
 25        CONTINUE

           DO 30 n=1,nmax
              c3(n)=1.0/(n+1.0)

              DO 27 l=0,lmax
                 c1(n,l)=2.0*n+twoalpha(l)
                 c2(n,l)=n-1.0+twoalpha(l)
 27           CONTINUE

 30        CONTINUE

        ENDIF

        lskip=1
        IF(zeroodd.OR.zeroeven) lskip=2
        lmin=0
        IF(zeroeven) lmin=1

C================== Calc pot and force ==================

         k=1
C        DO 200 k=1,nbodies-1

C           IF (mstpflag.AND.im(k).EQ.1) GO TO 200

crss     Following lines added for more efficient calculation of
crss     r, sin(theta), cos(theta), sin(phi), and cos(phi)

           tinynum = 1e-16 !gjy change: should not be less then 1e-16?? for xyz point (0.,0.,0.)
           !\ [warning] add softennings for nan, but changed x
           if( abs(x(k)) .le. tinynum ) then
             x(k) = sign(tinynum, x(k))
           endif
           if( abs(y(k)) .le. tinynum ) then
             y(k) = sign(tinynum, y(k))
           endif
           if( abs(z(k)) .le. tinynum ) then
             z(k) = sign(tinynum, z(k))
           endif
           rxyz2 = x(k)*x(k) + y(k)*y(k) + z(k)*z(k)
           rxy2 =  x(k)*x(k) + y(k)*y(k)
           arxyzinv(k) = 1.0/sqrt(rxyz2)
           arxyinv(k)  = 1.0/sqrt(rxy2)
           rxyzinv = arxyzinv(k)
           rxyinv  = arxyinv(k)
           r = rxyz2 * rxyzinv
           costh = z(k)*rxyzinv
           sinth = rxy2*rxyinv*rxyzinv
           cosmphi1 = x(k)*rxyinv
           sinmphi1 = y(k)*rxyinv
         !   arctanth = atan(z(k)*rxyzinv) !gjy add
           sin2thetaminv = -rxyz2/rxy2 !gjy add
         !   print*, "gd: xyzk: ", x(k), ", ", y(k), ", ", z(k)
         !   print*, "gd: cs : ", rxy2, ", ", costh, ", ", sin2thetaminv

crss     End new code

crss     Following lines commented out and replaced by more efficient
crss     calculations above
crss           r=SQRT(x(k)**2+y(k)**2+z(k)**2)
crss           costh=z(k)/r
crss           sinth=SQRT(1.-costh*costh)
crss           phi=ATAN2(y(k),x(k))
crss           cosmphi1=COS(phi)           
crss           sinmphi1=SIN(phi)           

           xi=(r-1.)/(r+1.)

           cosmphi(0)=1.0
           sinmphi(0)=0.
           cosmphi(1)=cosmphi1
           sinmphi(1)=sinmphi1

           DO 130 m=2,lmax
              cosmphi(m)=cosmphi1*cosmphi(m-1)-sinmphi1*sinmphi(m-1)
              sinmphi(m)=cosmphi1*sinmphi(m-1)+sinmphi1*cosmphi(m-1)
 130       CONTINUE
           adens(k)=0.0d0
           pot(k)=0.0d0
           ar=0.0d0
           ath=0.0d0
           aphi=0.0d0

           DO 148 l=0,lmax

              ultrasp(0,l)=1.0
              ultrasp(1,l)=twoalpha(l)*xi
              ultrasp1(0,l)=0.0
              ultrasp1(1,l)=1.0

              un=ultrasp(1,l)
              unm1=1.0

              DO 144 n=1,nmax-1
                 ultrasp(n+1,l)=(c1(n,l)*xi*un-c2(n,l)*unm1)*c3(n)
                 unm1=un
                 un=ultrasp(n+1,l)
                 ultrasp1(n+1,l)=((twoalpha(l)+n)*unm1-(n+1)*xi*
     &                    ultrasp(n+1,l))/(twoalpha(l)*(1.-xi*xi))
 144          CONTINUE

 148       CONTINUE

      ratio=1.

crss     Calculation of plm modified so as to avoid unnecessary logical
crss     tests for m.eq.0 inside loop over l, special case m=0 split off

           m = 0
           plm(m,m)=1.0
           plm1m=plm(m,m)
           plm2m=0.0
           DO l=m+1,lmax
              plm(l,m)=(costh*twolm1(l)*plm1m-
     %             lplusm(l,m)*plm2m)*lmm(l,m)
              plm2m=plm1m
              plm1m=plm(l,m)
           enddo

           DO m=1,lmax
              plm(m,m)=1.0
              ratio=-sinth*ratio
              plm(m,m)=dblfact(m)*ratio
              plm1m=plm(m,m)
              plm2m=0.0
              DO l=m+1,lmax
                 plm(l,m)=(costh*twolm1(l)*plm1m-
     %                lplusm(l,m)*plm2m)*lmm(l,m)
                 plm2m=plm1m
                 plm1m=plm(l,m)
              enddo
           enddo
crss     End modified loops for plm calculation

           dplm(0,0)=0.0

crss     Calculation of dplm modified so as to avoid unnecessary logical
crss     tests for l.eq.m inside loop over m, special case l=m split off

           DO l=1,lmax
               ! cos2theta_m1 = costh*costh-1.0 !gjy add
               ! if( cos2theta_m1 .le. 1e-28 ) then
               ! cos2theta_m1 = -1e-28
               ! else if( cos2theta_m1 .le. 1e-6 ) then
               ! cos2theta_m1 = -arctanth*arctanth
               ! endif
              DO m=0,l-1
                 dplm(l,m)=(l*costh*plm(l,m)-(l+m)*plm(l-1,m))*
     &                sin2thetaminv !gjy change
   !   &                (costh*costh-1.0) !gjy note: original
              enddo
              dplm(l,l)=l*costh*plm(l,m)*sin2thetaminv !gjy change
            !   dplm(l,l)=l*costh*plm(l,m)/(costh*costh-1.0) !gjy note: original
           enddo

crss     End modified loops for dplm calculation

           if(lmin.eq.0) then
           phinltil = 1./(1+r)
           else
           phinltil=r/(1.+r**3)
           endif
           ratio=r/(1.+r)**2
           if(lskip.eq.2) ratio=ratio*ratio
           DO 190 l=lmin,lmax,lskip

              temp2=0.0
              temp3=0.0
              temp4=0.0
              temp5=0.0
              temp6=0.0

              DO 180 m=0,l,2
C              DO 180 m=0,0

                 alm=0.0
                 blm=0.0
                 clm=0.0
                 dlm=0.0
                 elm=0.0
                 flm=0.0

                 DO 150 n=0,nmax
                    alm=alm+knl(n,l)*ultrasp(n,l)*cossum(n,l,m)
                    blm=blm+knl(n,l)*ultrasp(n,l)*sinsum(n,l,m)
                    clm=clm+ultrasp(n,l)*cossum(n,l,m)
                    dlm=dlm+ultrasp(n,l)*sinsum(n,l,m)
                    elm=elm+ultrasp1(n,l)*cossum(n,l,m)
                    flm=flm+ultrasp1(n,l)*sinsum(n,l,m)
 150             CONTINUE

		 temp2=temp2+plm(l,m)*(alm*cosmphi(m)+blm*sinmphi(m))
                 temp3=temp3+plm(l,m)*(clm*cosmphi(m)+dlm*sinmphi(m))
                 temp4=temp4-plm(l,m)*(elm*cosmphi(m)+flm*sinmphi(m))
                 temp5=temp5-dplm(l,m)*(clm*cosmphi(m)+dlm*sinmphi(m))
                 temp6=temp6-m*plm(l,m)*(dlm*cosmphi(m)-clm*sinmphi(m))
 180          CONTINUE


	      rhonltil=phinltil/(r*(1.+r)*(1.+r)*2.0d0*pi)
CC
              pot(k)=pot(k)+temp3*phinltil
CC
CC
	      adens(k)=adens(k)-temp2*rhonltil
CC
              ar=ar+phinltil*(-temp3*(l/r-twolp1(l)/
     &              (1.+r))+temp4*2.*twoalpha(l)/(1.+r)**2)
              ath=ath+temp5*phinltil
              aphi=aphi+temp6*phinltil
           phinltil = phinltil*ratio
 190       CONTINUE

           ath= -sinth*ath/r
           aphi=aphi/(r*sinth)
           ax(k)=G*(sinth*cosmphi1*ar+costh*cosmphi1*ath-
     &           sinmphi1*aphi)
           ay(k)=G*(sinth*sinmphi1*ar+costh*sinmphi1*ath+
     &           cosmphi1*aphi)
           az(k)=G*(costh*ar-sinth*ath)
           pot(k)=pot(k)*G

C++++++++++++++++++++++++++++++++++++++++++++++++++++++
C              Add BH's contribution
C++++++++++++++++++++++++++++++++++++++++++++++++++++++
C        mbh = mass(nbodies)
C        pot(k) = pot(k) - mbh*G*rxyzinv
        
C        rxyzinv3 = rxyzinv*rxyzinv*rxyzinv
C        ax(k) = ax(k) - mbh*G*x(k)*rxyzinv3
C        ay(k) = ay(k) - mbh*G*y(k)*rxyzinv3
C        az(k) = az(k) - mbh*G*z(k)*rxyzinv3
        
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
CC
C    NB no convective derivative!
CC
C	 dax(k)=-4.*pi*G*adens(k)*vx(k) 
C	 day(k)=-4.*pi*G*adens(k)*vy(k) 
C	 daz(k)=-4.*pi*G*adens(k)*vz(k) 

C 200    CONTINUE

C        CALL out_pot
C        CALL out_force

        RETURN
        END
C***********************************************************************
C***********************************************************************
C
C
                     SUBROUTINE iocoef(sinsum,cossum)
C
C
C***********************************************************************
C
C
C     Subroutine to input and output expansion coefficients.
C
C
C=======================================================================

        INCLUDE 'tmhscf.h'   
C        INCLUDE 'mpif.h'

        INTEGER n,l,m
        LOGICAL firstc
        REAL*8 sinsum,cossum,tt, mbh

        DIMENSION sinsum(0:nmax,0:lmax,0:lmax),
     &            cossum(0:nmax,0:lmax,0:lmax)

        DATA firstc/.TRUE./

        SAVE firstc
        
        COMMON/BH/mbh

C=======================================================================

C        tmptime = mpi_wtime()

        IF(firstc) THEN

           firstc=.FALSE.

           IF(outpcoef .and. me.eq.0)
     &          OPEN(uoutcoef,FILE=outcfile,STATUS='UNKNOWN')
           IF(inptcoef .and. me.eq.0)
     &          OPEN(uincoef,FILE=incfile,STATUS='OLD')

        ENDIF

        IF(outpcoef .and. me.eq.0) THEN

           WRITE(uoutcoef,100) tnow, mbh

           DO 30 n=0,nmax
              DO 20 l=0,lmax
                 DO 10 m=0,l
                    WRITE(uoutcoef,100) sinsum(n,l,m),cossum(n,l,m)
 10              CONTINUE
 20           CONTINUE
 30        CONTINUE

 100       FORMAT(1x,10(1pe22.13))

        ENDIF

        IF(inptcoef) THEN
            REWIND(uincoef)
Cgjy add: rewind
           if (me .eq. 0) then
                READ(uincoef,*) tt, mbh
                tmpsum01 = tt
           endif
C           call share(tt,1)
C           write(*,*) 'Reading coeff...'
C           write(*,"('t   = ',1pe11.3)") tt
C           write(*,"('mbh = ',1pe11.3)") mbh
           

C           IF(tt.NE.tnow) CALL terror(' input error in iocoef ')
C                              ------

           DO 130 n=0,nmax
              DO 120 l=0,lmax
                 DO 110 m=0,l
                    if (me .eq. 0) then
                      READ(uincoef,*) sinsum(n,l,m),cossum(n,l,m)
                    end if
C                    call share(sinsum(n,l,m),1)
C                    call share(cossum(n,l,m),1)
 110             CONTINUE
 120          CONTINUE
 130       CONTINUE

        ENDIF

C        tmptime = mpi_wtime()-tmptime
C        cputime0 = cputime0 + tmptime
C        cputime = cputime + tmptime

        RETURN
        END
C***********************************************************************




C***********************************************************************
C
C
                          SUBROUTINE initpars
C
C
C***********************************************************************
C
C
C     Subroutine to initialize system parameters that depend on
C     either the input data or defined PARAMETERS.
C
C
C=======================================================================

        INCLUDE 'tmhscf.h'  
C        include 'mpif.h' 

C=======================================================================

C   Initialize misc. useful numbers.
C   --------------------------------
        one=1.0
        two=2.0
        pi=4.0*ATAN(one)
        twoopi=2./pi
        onesixth=1./6.
        tiny=1.e-30
        zero=0.0

        tpos=tnow
        tvel=tnow

        dtime = 1.e-6
        dtsmall = 1.e-6
        dtbig = 1.e-6
        tnextbig= tnow + dtbig
        mstpflag=.false.
        
        me = 0

        RETURN
        END

C***********************************************************************
C
C
                          SUBROUTINE inparams
C
C
C***********************************************************************
C
C
C     Subroutine to read in parameters.
C
C     Input parameters:
C
C        headline  : identification string for the run.
C        nsteps    : number of timesteps.
C        noutbod   : output system state once every nsteps/noutbod 
C                    steps.
C        noutlog   : output logfile data once every nsteps/noutlog
C                    steps.
C        dtime     : the timestep.
C        G         : value of gravitational constant, in appropriate
C                    units.
C        selfgrav  : option to turn off (.FALSE.) system self-gravity.
C        inptcoef  : option to read-in expansion coefficients.
C        outpcoef  : option to write-out expansion coefficients.
C        zeroodd   : option to zero all odd terms in the expansion.
C        zeroeven  : option to zero all even terms in the expansion.
C        fixacc    : option to force conservation of linear
C                    momentum by subtracting acceleration of c.o.m.
C
C	ecrit	   : energy below which we want particle to be n^2 integrated
C	rcrit	   : radius below which we want particle to be multistepped
C		    as multiple of radbh (typically 0.5-1)
C=======================================================================

        INCLUDE 'tmhscf.h'   

        CHARACTER *1 pcomment

C=======================================================================

        IF (me .eq. 0) THEN
            OPEN(UNIT=upars,FILE=parsfile,STATUS='OLD')

C   Read parameters, close the file.
C   --------------------------------

            READ(upars,'(a)') pcomment

            READ(upars,'(a)') headline
            READ(upars,*) nsteps
            READ(upars,*) noutbod
            READ(upars,*) noutlog
            READ(upars,*) dteps
            READ(upars,*) G
            READ(upars,*) tfinal
            READ(upars,*) multistep
            READ(upars,*) fixedn
            READ(upars,*) selfgrav
            READ(upars,*) inptcoef
            READ(upars,*) outpcoef
            READ(upars,*) zeroodd
            READ(upars,*) zeroeven
            READ(upars,*) fixacc
            READ(upars,*) rcrit
            READ(upars,*) ecrit
	    READ(upars,*) lilout
	    READ(upars,*) nlilout

            CLOSE(UNIT=upars)

C***********************************************************************
C
C        inparams now reads in SCFMOD with other data
C        Steinn Sigurdsson, Dec. 1993
C
C***********************************************************************
C
C
C     Subroutine to read in parameters.
C
C     Input parameters:
C
C        pcomment    : first line
C        iseed       : initialises random number generator
C       msys        : mass of system to be generated
C       rsys        : radius of system
C       r0          : scale radius
C       bhmass      : mass of black hole
C       epsbh       : softening length for black hole
C       tstartbh    : time to start bh growth
C       tgrowbh     : time to grow bh
C       tlivebh     : time bh lives for
C       tdiebh      : time to shrink bh
C       xdrag       : drag coeff for vx
C       ydrag       : drag coeff for vy
C       zdrag       : drag coeff for vz
C       tstartd  : time drag starts
C        tgrowdrag   : time drag lasts
C        tdiedrag    : time drag dies down
C        mkmod       : make internal model or read in data?
C       bhgrav      : do we grow a black hole
C       bhgrav      : do we grow a black hole
C       usedrag     : is there drag
C       stellev     : is there stellar evolution
C
C=======================================================================
            OPEN(UNIT=umods,FILE=modsfile,STATUS='OLD')

            READ(umods,'(a)')
            READ(umods,*)iseed
            READ(umods,*)bhmass
            READ(umods,*)epsbh
            READ(umods,*)tstartbh
            READ(umods,*)tgrowbh
            READ(umods,*)tlivebh
            READ(umods,*)tdiebh
            READ(umods,*)xdrag
            READ(umods,*)ydrag
            READ(umods,*)zdrag
            READ(umods,*)tstartdrag
            READ(umods,*)tgrowdrag
            READ(umods,*)tlivedrag
            READ(umods,*)tdiedrag
            READ(umods,*)bhgrav
            READ(umods,*)usedrag
            READ(umods,*)stellev
C
            CLOSE(UNIT=umods)
C
	ENDIF

        dtime = dteps
        nfrac = 1

        RETURN
        END
C***********************************************************************

C***********************************************************************
C
C
        FUNCTION FACTRL(N)
C
C
C***********************************************************************
C
C
C     A function to compute factorials.  (From numerical recipes.)
C
C
C=======================================================================

        INTEGER n,ntop,j
        REAL*8 factrl,a,gammln,arggam

        DIMENSION A(33)

        DATA NTOP,A(1)/0,1./

        IF (N.LT.0) THEN
          PAUSE 'negative factorial'
        ELSE IF (N.LE.NTOP) THEN
          FACTRL=A(N+1)
        ELSE IF (N.LE.32) THEN
          DO 11 J=NTOP+1,N
            A(J+1)=J*A(J)
11        CONTINUE
          NTOP=N
          FACTRL=A(N+1)
        ELSE
          arggam=n+1.
          FACTRL=EXP(GAMMLN(arggam))
        ENDIF

        RETURN
        END
C***********************************************************************
C
C
        FUNCTION GAMMLN(XX)
C
C
C***********************************************************************
C
C
C     A routine to compute the natural logarithm of the gamma
C     function.  (Taken from numerical recipes.)
C
C
C=======================================================================

        INTEGER j

        REAL*8 COF(6),STP,HALF,ONE,FPF,X,TMP,SER,gammln,xx

        DATA COF,STP/76.18009173D0,-86.50532033D0,24.01409822D0,
     &      -1.231739516D0,.120858003D-2,-.536382D-5,2.50662827465D0/
        DATA HALF,ONE,FPF/0.5D0,1.0D0,5.5D0/

        X=XX-ONE
        TMP=X+FPF
        TMP=(X+HALF)*LOG(TMP)-TMP
        SER=ONE

        DO 11 J=1,6
          X=X+ONE
          SER=SER+COF(J)/X
11      CONTINUE

        GAMMLN=TMP+LOG(STP*SER)
        
        RETURN
        END
C***********************************************************************


