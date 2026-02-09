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
	! subroutine get_pot( xi, poti )
	! include 'tmhscf.h'
    
	! real*8 mbh, eps, r1, xi(3), poti
    
	! common /BH/ mbh
	! common /EPS/ eps
    
	! eps = 1.0000D-05
    
	! r1 = xi(1)**2. + xi(2)**2. + xi(3)**2. + eps*eps
	! r1 = sqrt(r1)
    
	! x(1) = xi(1); y(1) = xi(2); z(1) = xi(3)
	! call calc_a
	
	! poti = pot(1) - mbh/r1
    
	! return
	! end
C######################################################

C######################################################
	subroutine get_pot( xi, poti )
		!gjy note: when xcar_target is like {1e-8, 1e-8, 1.} 
		!\ (1e-7 is not), nan will occur in the original prog, 
		!\ so one uses interpolation to replace nan.

		use,intrinsic :: IEEE_ARITHMETIC !gjy add: for IEEE_IS_NaN
		include 'tmhscf.h'
		
		real*8 mbh, eps, r1, xi(3), poti
		real*8 xs(4), ys(4), xt, yt, yxt, ri, rxs(4) 
		!\gjy note: xitmp(3) is x(3) in 'tmhscf.h' called by cal_a, pot_tmp is pot
		real*8, parameter::Err3 = 1.e-2 !const !gjy add
		
		common /BH/ mbh
		common /EPS/ eps
		
		eps = 1.0000D-05
		
		r1 = xi(1)**2. + xi(2)**2. + xi(3)**2. + eps*eps
		r1 = sqrt(r1)
		
		x(1) = xi(1); y(1) = xi(2); z(1) = xi(3)
		call calc_a
		poti = pot(1) - mbh/r1

		! print*, "NaN poti = ", poti !gjy add
		! if( IEEE_IS_NaN(poti) ) then
		! 	rxs(1) = -1e-1; rxs(2) = -1e-2; rxs(3) = 1e-2; rxs(4) = 1e-1
		! 	!\gjy note: change to near xi(3) in a line??
		! 	ri = sqrt(xi(1)**2. + xi(2)**2. + xi(3)**2.)
		! 	do i = 1,4
		! 		do j = 1,3
		! 			x(j) = rxs(i)
		! 		end do
		! 		xs(i) = sqrt(x(1)**2. + x(2)**2. + x(3)**2.)*x(1)/abs(x(1))
		! 		call calc_a
		! 		ys(i) = pot(1)
		! 	end do

		! 	xt = ri
		! 	call interpolate_3o_spline_1d( xs, ys, 4, xt, yt, yxt, 1 )
		! 	poti = yt
		! 	print*, "xs: ", xs
		! 	print*, "ys: ", ys
		! 	print*, "xt: ", xt
		! 	print*, "yxt: ", yxt
		! end if

		return
	end
C######################################################
