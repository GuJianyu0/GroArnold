C######################################################
C    File   : force_SCF.f
C    Func.  : Using SCF code to calc force and pot,
C           : subroutines in this file provide a interface to 
C           : SCF functions in 'SCF-code.f'
C    Start  : 2013-11-17, 17:00
C######################################################


C######################################################
	subroutine get_parameter
	include 'tmhscf.h'
    
	real*8 mbh, rtmp
	real*8 sinsum(0:nmax,0:lmax,0:lmax)
	real*8 cossum(0:nmax,0:lmax,0:lmax)
	common/BH/mbh
	common/sincos/sinsum,cossum
    
	CALL inparams
	CALL initpars
    
	inptcoef = .TRUE.
	outpcoef = .FALSE.
	sinsum = 0.0
	cossum = 0.0
	call iocoef( sinsum, cossum )
	
	return
	end
C######################################################


C######################################################
Csubroutine check_parameter()
C    implicit none
C    integer lmax, l, itmp
C    real*8 alpha(0:19), R, coeff(6,0:50)
C    real*8 mbh, rtmp
C    common /param/ lmax, alpha, R, coeff
C    common /MBH/ mbh
    
C    print*, lmax, R, mbh
C    do l=0,lmax
C      write(*,*) coeff(1:6,l)
C    end do
    
Cend subroutine check_parameter
C######################################################



C######################################################
	subroutine get_pot_xyz( xi, yi, zi, poti )
	include 'tmhscf.h'
    
	real*8 mbh, eps, r1, xi, yi, zi, poti
    
	common /BH/ mbh
	common /EPS/ eps
    
	eps = 1.0000D-05
    
	r1 = xi**2. + yi**2. + zi**2. + eps*eps
	r1 = sqrt(r1)
    
	x(1) = xi; y(1) = yi; z(1) = zi
	call calc_a
	
	poti = pot(1) - mbh/r1
    
	return
	end
C######################################################

C######################################################
	subroutine get_acc_xyz( xi, yi, zi, axi, ayi, azi )
	include 'tmhscf.h'
    
	real*8 mbh, eps, r1, xi, yi, zi, axi, ayi, azi
    
	common /BH/ mbh
	common /EPS/ eps
    
	eps = 1.0000D-05
    
	r1 = xi**2. + yi**2. + zi**2. + eps*eps
	r1 = sqrt(r1)
    
	x(1) = xi; y(1) = yi; z(1) = zi
	call calc_a
	
	! poti = pot(1) - mbh/r1
	axi = ax(1) - mbh*xi/r1**3.
	ayi = ay(1) - mbh*yi/r1**3.
	azi = az(1) - mbh*zi/r1**3.
    
	return
	end
C######################################################



C######################################################
C gjy add: to return 3 dim pot (m=1)
	subroutine get_pot( xi, poti )
	include 'tmhscf.h'
    
	real*8 mbh, eps, r1, xi(3), poti
    
	common /BH/ mbh
	common /EPS/ eps
    
	eps = 1.0000D-05
    
	r1 = xi(1)**2. + xi(2)**2. + xi(3)**2. + eps*eps
	r1 = sqrt(r1)
    
	x(1) = xi(1); y(1) = xi(2); z(1) = xi(3)
	call calc_a
	
	poti = pot(1) - mbh/r1
    
	return
	end
C######################################################

C######################################################
C gjy add: to return 3 dim forces (m=1)
	subroutine get_acc( xi, yi, zi, axi, ayi, azi )
	include 'tmhscf.h'
    
	real*8 mbh, eps, r1, xi, yi, zi, axi, ayi, azi
    
	common /BH/ mbh
	common /EPS/ eps
    
	eps = 1.0000D-05
    
	r1 = xi**2. + yi**2. + zi**2. + eps*eps
	r1 = sqrt(r1)
    
	x(1) = xi; y(1) = yi; z(1) = zi
	call calc_a
	
	! poti = pot(1) - mbh/r1
	axi = ax(1) - mbh*xi/r1**3.
	ayi = ay(1) - mbh*yi/r1**3.
	azi = az(1) - mbh*zi/r1**3.
    
	return
	end
C######################################################

Cvecdot
C######################################################
	subroutine get_force( xi, vi, ai, adoti )
	include 'tmhscf.h'
    
	real*8 xi(3), vi(3), ai(3), adoti(3)
	real*8 r1, r2, mbh, vdotr, eps
	integer i
	common /BH/ mbh
	common /EPS/ eps
    
	call vector_dot( xi, xi, r2 )
	r2 = r2 + eps*eps
	r1 = dsqrt(r2)
    
	x(1) = xi(1)
	y(1) = xi(2)
	z(1) = xi(3)
	call calc_a
	ai(1) = ax(1)
	ai(2) = ay(1)
	ai(3) = az(1)
    
C	call calc_adot( xi, vi, ai, adoti )
	call calc_adot_2( xi, vi, ai, adoti )
    
	call vector_dot( xi, xi, r2 )
	r2 = r2 + eps*eps
	r1 = dsqrt(r2)
	
	do i=1,3
	  ai(i) = ai(i) - xi(i) * mbh / (r1*r2)
	end do
	call vector_dot(xi, vi, vdotr)
	do i=1,3
	  adoti(i) = adoti(i) - mbh*(vi(i)/(r1*r2)
     &             - 3.0*vdotr*xi(i)/(r1*r2*r2))
	end do
	return
	end
C######################################################


C######################################################
	subroutine calc_adot( xi, vi, ai, adoti )
	include 'tmhscf.h'
    
	integer i, j

	real*8 xi(3), vi(3), ai(3), adoti(3)
	real*8 xnew(3), anew(3)
	real*8 r1, r2
	real*8 a_diff(3,3), h
    
	h = 1.0000D-07
	adoti = 0.0000D+00
    
C  /** Calc a_diff **/
C  /** a_diff(i,j) = d_a(j)/d_x(i) ,where d_x(i)==h  **/
	do i=1,3
	  xnew = xi
	  xnew(i) = xnew(i) + h
	
	  x(1) = xnew(1)
	  y(1) = xnew(2)
	  z(1) = xnew(3)
      
	  call calc_a
	  anew(1) = ax(1)
	  anew(2) = ay(1)
	  anew(3) = az(1)
      
	  do j=1,3
	    a_diff(i,j) = anew(j) - ai(j)
	  end do
	end do
    
	do i=1,3
	  do j=1,3
	     a_diff(i,j) = a_diff(i,j)/h
	  end do
	end do
C   /** Calc adot **/
	do i=1,3
	  do j=1,3
	    adoti(j) = adoti(j) + vi(i)*a_diff(i,j)
	  end do
	end do

	return
	end
C######################################################

C######################################################
	subroutine calc_adot_2( xi, vi, ai, adoti )
	include 'tmhscf.h'
	
	integer i,j
	
	real*8 xi(3), vi(3), ai(3), adoti(3), a_diff(3,3)
	real*8 xfwd(3), xbkw(3), afwd(3), abkw(3)
	real*8,parameter:: h = 1.0D-07
	  
C a_diff(i,j)=d_a(j)/d_x(i), where d_x(i)=h
	do i=1,3

	  xfwd = xi
	  xfwd(i) = xfwd(i) + h
	  x(1) = xfwd(1)
	  y(1) = xfwd(2)
	  z(1) = xfwd(3)
	  call calc_a
	  afwd(1) = ax(1)
	  afwd(2) = ay(1)
	  afwd(3) = az(1)

	  xbkw = xi
	  xbkw(i) = xbkw(i) - h
	  x(1) = xbkw(1)
	  y(1) = xbkw(2)
	  z(1) = xbkw(3)
	  call calc_a
	  abkw(1) = ax(1)
	  abkw(2) = ay(1)
	  abkw(3) = az(1)

	
	  do j=1,3
	    a_diff(i,j) = afwd(j) - abkw(j)
	  end do ! j loop
	end do ! i loop

	do i=1,3
	  do j=1,3
	    a_diff(i,j) = a_diff(i,j)/(2.*h)
	  end do ! j loop
	end do ! i loop
	
	adoti = 0.0
	do i=1,3
	  do j=1,3
	    adoti(j) = adoti(j) + vi(i)*a_diff(i,j)
	  end do ! j loop
	end do ! i loop
	
	return
	end
!######################################################



Cvecdot
C######################################################
	subroutine energy_contr( x, v, e_kin, e_pot, e_tot )
	real*8 x(3), v(3), e_kin, e_pot, e_tot
	real*8 v2
    
	call get_pot( x, e_pot )
	call vector_dot( v, v, v2 )
    
	e_kin = 0.5 * v2
	e_tot = e_kin + e_pot
    
	return
	end
C######################################################



C######################################################
C   Only root process can call this subroutine !!!
C######################################################

	subroutine ReadCoeff( sinlist, coslist, mmm, zo, ze, gconst )

C######################################################

	include 'tmhscf.h'
	
	integer i, n, l, m
	
	logical zo, ze

	real*8 sinsum(0:nmax,0:lmax,0:lmax)
	real*8 cossum(0:nmax,0:lmax,0:lmax)
    
	real*8 mbh, rtmp, gconst, mmm
	real*8 sinlist(0:4095), coslist(0:4095)
	
	common/BH/mbh
CC	common/sincos/sinsum,cossum
    
	sinlist = 0.0
	coslist = 0.0
    
	CALL inparams
CC	CALL initpars
	
	inptcoef = .TRUE.
	outpcoef = .FALSE.
	call iocoef(sinsum,cossum)
    
	i=0
	do n=0,nmax
	  do l=0,lmax
            do m=0,l
          
              sinlist(i) = sinsum(n,l,m)
              coslist(i) = cossum(n,l,m)
              i = i + 1
          
            end do
          end do
	end do
	
	mmm = mbh
	gconst = G
	zo = zeroodd
	ze = zeroeven
    
	return
	end
C######################################################



C######################################################

	subroutine set_parameter( sinlist, coslist, mmm, zo, ze, gconst )

C######################################################
	include 'tmhscf.h'
	
	integer i, n, l, m
	
	logical zo, ze

	real*8 sinsum(0:nmax,0:lmax,0:lmax)
	real*8 cossum(0:nmax,0:lmax,0:lmax)
    
	real*8 mbh, rtmp, gconst, mmm
	real*8 sinlist(0:4095), coslist(0:4095)
	
	common/BH/mbh
	common/sincos/sinsum,cossum
	
	CALL initpars
	
	i=0
	do n=0,nmax
	  do l=0,lmax
            do m=0,l
          
              sinsum(n,l,m) = sinlist(i)
              cossum(n,l,m) = coslist(i)
              i = i + 1
          
            end do
          end do
	end do
	
	mbh = mmm
	G = gconst
	zeroodd = zo
	zeroeven = ze
	
	return
	end
C######################################################


