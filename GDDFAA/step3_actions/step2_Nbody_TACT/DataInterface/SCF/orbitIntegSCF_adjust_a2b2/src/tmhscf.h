C=======================================================================
C
C
C                        INCLUDE FILE tmhscf.h
C
C
C=======================================================================
C
C
C     Parameter declarations, allocation of array storage, common
C     block definitions.
C
C
C=======================================================================

C=======================================================================
C     Definitions I added -- Bohr He, June 1995
C     Further Modifications by Steinn Sigurdsson Sep 1995
C=======================================================================
        integer n_pes, me, lognpes
        real *8 tmpsum01, tmpsum02, tmpsum03, tmpsum04, tmpsum05,
     &          tmpsum06, tmpsum07, tmpsum08, tmpsum09, tmpsum10,
     &          tmpsum11, tmpsum12, tmpsum13, tmpsum14
        real *8 temp01, temp02, temp03, temp04, temp05, temp06, temp07,
     &          temp08, temp09, temp10,temp11
        integer itemp01
        real *8 totime0,totime1,totime,tmptime,cycletime
        parameter(cycletime=6.6666666667E-9)
C=======================================================================

        INTEGER nbodsmax,nbodsper,nmax,lmax
 
C        PARAMETER(nbodsmax=2048000,nbodsper=2048000,nmax=42,lmax=15)
Cgjy change: wrong scficoef
C        PARAMETER(nbodsmax=2048000,nbodsper=2048000,nmax=70,lmax=15)
C        PARAMETER(nbodsmax=2048000,nbodsper=2048000,nmax=30,lmax=10)
        PARAMETER(nbodsmax=2048000,nbodsper=2048000,nmax=14,lmax=10)
Cgjy changed: original 14 10 10
C        PARAMETER(nbodsmax=2048000,nbodsper=2048000,nmax=4,lmax=2)
Cgjy changed: brute force smoothing

        CHARACTER*50 headline
        INTEGER nsteps,noutbod,nbodies,noutlog,nfrac,nlilout,iseed
	INTEGER im

        LOGICAL selfgrav,inptcoef,outpcoef,zeroodd,zeroeven,fixacc
        LOGICAL bhgrav,multistep,usedrag,stellev,fixedn,lilout
	LOGICAL mstpflag

        REAL*8 tnow,x,y,z,vx,vy,vz,mass,pot,dtime,G,ax,ay,az,one,pi,
     &         twoopi,onesixth,tpos,tvel,cputime0,cputime1,cputime,
     &         rcrit,radbh,vel0bh,dteps,
     &         potext,two,zero,tiny,ecrit,oax,oay,oaz,odax,oday,odaz,
     &         dax,day,daz,adens,dti
        REAL*8 bhmass,epsbh,tstartbh,tgrowbh,tlivebh,tdiebh,
     &         tstartdrag,tgrowdrag,
     &         xdrag,ydrag,zdrag,tlivedrag,tdiedrag,bhmasst,tfinal

	REAL*8 sinsum1,sinsum2,cossum1,cossum2
	REAL*8 dts,dtlil
        REAL*8 dtsmall,dtbig,tnextbig

        COMMON/bodscom/x(nbodsper),y(nbodsper),z(nbodsper),vx(nbodsper),
     &                 vy(nbodsper),vz(nbodsper),mass(nbodsper),
     &                 pot(nbodsper),ax(nbodsper),ay(nbodsper),
     &                 az(nbodsper),potext(nbodsper),adens(nbodsper),
     &                 dax(nbodsper),day(nbodsper),daz(nbodsper),
     &                 oax(nbodsper),oay(nbodsper),oaz(nbodsper),
     &                 odax(nbodsper),oday(nbodsper),odaz(nbodsper),
     &                 dti(nbodsper)
        COMMON/parcomi/nbodies,nsteps,noutbod,noutlog,nlilout,nfrac
        COMMON/parcomr/dtime,G,one,pi,twoopi,onesixth,two,tiny,zero
        COMMON/parcomc/headline
        COMMON/parcoml/selfgrav,inptcoef,outpcoef,zeroodd,zeroeven,
     &           lilout,fixacc,bhgrav,multistep,usedrag,stellev,fixedn
        COMMON/timecom/tpos,tnow,tvel,ecrit,dteps
	COMMON/timecom2/rcrit,radbh,vel0bh,tfinal
	COMMON/tstepcom/dts,dtlil
        COMMON/cpucom/cputime0,cputime1,cputime
        COMMON/bhgcom/bhmass,epsbh,tstartbh,tgrowbh,tlivebh,tdiebh
        COMMON/dracom/xdrag,ydrag,zdrag,tlivedrag,tdiedrag,
     &                tstartdrag,tgrowdrag,bhmasst
	COMMON/coefcom/sinsum1(0:nmax,0:lmax,0:lmax),
     &                 sinsum2(0:nmax,0:lmax,0:lmax),
     &                 cossum1(0:nmax,0:lmax,0:lmax),
     &                 cossum2(0:nmax,0:lmax,0:lmax)
        COMMON/mstpcomr/dtsmall,dtbig,tnextbig
        COMMON/mstpcomi/im(nbodsper)
        COMMON/mstpcoml/mstpflag


C=======================================================================
C     Definition I added -- Bohr He, June 1995
C=======================================================================
        common/t3d/n_pes, me, lognpes
        common/tmpo/tmpsum01, tmpsum02, tmpsum03, tmpsum04, tmpsum05,
     &              tmpsum06, tmpsum07, tmpsum08, tmpsum09, tmpsum10,
     &              tmpsum11, tmpsum12, tmpsum13, tmpsum14
        common/tempo/temp01(nbodsper),temp02(nbodsper),temp03(nbodsper),
     &               temp04(nbodsper),temp05(nbodsper),temp06(nbodsper),
     &               temp07(nbodsper),temp08(nbodsper),temp09(nbodsper),
     &               temp10(nbodsper),temp11(nbodsper)
        common/itempo/itemp01(nbodsper)
        common/cpucom1/totime0,totime1,totime,tmptime

C=======================================================================
C   Definitions specific to input/output.
C=======================================================================
        INTEGER uterm,upars,ulog,ubodsin,ubodsout,utermfil,uoutcoef,
     &          uincoef,ubodsel,umods,ubodslil,uchkout
        CHARACTER*8 parsfile,logfile,ibodfile,obodfile,termfile,
     &              outcfile,incfile,elfile,modsfile,olilfile,
     &              chkfile

        PARAMETER(uterm=6,upars=10,ulog=11,ubodsin=12,ubodsout=13,
     &            umods=14,utermfil=15,uoutcoef=16,uincoef=17,
     &            ubodsel=18,ubodslil=19,uchkout=20)
        PARAMETER(parsfile='scfpar',logfile='scflog',
     &            modsfile='scfmod',olilfile='slilxxxx',
     &        ibodfile='data.inp',obodfile='snapxxxx',
     &            termfile='scfout',outcfile='scfocoef',
     &            incfile='scficoef',elfile='scfelxxx',
     &            chkfile='scfchkpt')




