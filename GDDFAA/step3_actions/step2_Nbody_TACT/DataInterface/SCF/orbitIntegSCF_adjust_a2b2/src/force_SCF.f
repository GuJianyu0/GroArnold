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

	subroutine get_force( xi, vi, ai, adoti ) !gjy change: for IEEE_IS_NaN
		use,intrinsic :: IEEE_ARITHMETIC !gjy add: for IEEE_IS_NaN
		include 'tmhscf.h'
		
		real*8 xi(3), vi(3), ai(3), adoti(3)
		real*8 r1, r2, mbh, vdotr, eps
		real*8 xitmp(3), vitmp(3), aitmp(3), adotitmp(3)
		real*8 tgnl !gjy add: to goto new line
		real*8, parameter::Err3 = 1.e-2 !const !gjy add
		integer i
		common /BH/ mbh
		common /EPS/ eps
		! print*, "eps: ", eps !gjy add: 1e-5
		
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
		
		! call calc_adot( xi, vi, ai, adoti )
		call calc_adot_2( xi, vi, ai, adoti )
		
		call vector_dot( xi, xi, r2 )
		r2 = r2 + eps*eps
		r1 = dsqrt(r2)
		
		do i=1,3
		ai(i) = ai(i) - xi(i) * mbh / (r1*r2)
		end do
		call vector_dot(xi, vi, vdotr)
		do i=1,3
			tgnl = mbh*(vi(i)/(r1*r2)-3.0*vdotr*xi(i)/(r1*r2*r2))
			adoti(i) = adoti(i)-tgnl
			! gjy change: code new line ??
	  		! adoti(i) = adoti(i)-mbh*(vi(i)/(r1*r2)-3.0*vdotr*xi(i)/(r1*r2*r2))
			! &             - 3.0*vdotr*xi(i)/(r1*r2*r2))
		end do
		
		! do i=1,3 !gjy add: NaN is usually occur when xyz near (0.,0.,0.), so change NaN to ??
		! 	if( IEEE_IS_NaN(ai(i)) ) then
		! 		print*,"NaN ai i=",i
		! 		! ai(i) = -xi(i)*Err3
		! 		ai(i) = 123
		! 	end if
		! end do
		! do i=1,3 !gjy add
		! 	if( IEEE_IS_NaN(adoti(i)) ) then
		! 		print*,"NaN adoti i=",i
		! 		! ai(i) = xi(i)*Err3
		! 		ai(i) = 1234
		! 	end if
		! end do
		
		return
	end
C######################################################


! C######################################################
! subroutine get_force( xi, vi, ai, adoti )
! 	! use,intrinsic :: IEEE_ARITHMETIC !gjy add: for IEEE_IS_NaN
! 	include 'tmhscf.h'
    
! 	real*8 xi(3), vi(3), ai(3), adoti(3)
! 	real*8 r1, r2, mbh, vdotr, eps
! 	real*8 xitmp(3), vitmp(3), aitmp(3), adotitmp(3)
! 	real*8, parameter::Err3 = 1.e-2 !const !gjy add
! 	integer i
! 	common /BH/ mbh
! 	common /EPS/ eps
    
! 	call vector_dot( xi, xi, r2 )
! 	r2 = r2 + eps*eps
! 	r1 = dsqrt(r2)
    
! 	x(1) = xi(1)
! 	y(1) = xi(2)
! 	z(1) = xi(3)
! 	call calc_a
! 	ai(1) = ax(1)
! 	ai(2) = ay(1)
! 	ai(3) = az(1)
    
! C	call calc_adot( xi, vi, ai, adoti )
! 	call calc_adot_2( xi, vi, ai, adoti )
    
! 	call vector_dot( xi, xi, r2 )
! 	r2 = r2 + eps*eps
! 	r1 = dsqrt(r2)
	
! 	do i=1,3
! 	  ai(i) = ai(i) - xi(i) * mbh / (r1*r2)
! 	end do
! 	call vector_dot(xi, vi, vdotr)
! 	do i=1,3
! 	  adoti(i) = adoti(i) - mbh*(vi(i)/(r1*r2)
!      &             - 3.0*vdotr*xi(i)/(r1*r2*r2))
! 	end do
	
! 	! do i=1,3 !gjy add: NaN is usually occur when xyz near (0.,0.,0.), so change NaN to ??
! 	! 	if( IEEE_IS_NaN(ai(i)) ) then
! 	! 		print*,"NaN ai i=",i
! 	! 		! ai(i) = -xi(i)*Err3
! 	! 		ai(i) = 123
! 	! 	end if
! 	! end do
! 	! do i=1,3 !gjy add
! 	! 	if( IEEE_IS_NaN(adoti(i)) ) then
! 	! 		print*,"NaN adoti i=",i
! 	! 		! ai(i) = xi(i)*Err3
! 	! 		ai(i) = 1234
! 	! 	end if
! 	! end do
	
! 	return
! 	end
! C######################################################

! C######################################################
! 	subroutine get_force( xi, vi, ai, adoti ) !gjy add
! 		implicit none
! 		include 'tmhscf.h' !??
! 		real*8 xi(3), vi(3), ai(3), adoti(3)
! 		real*8 x1, y1, z1, ax, ay, az, accx, accy, accz
! 		real*8, parameter::Err1 = 1.e-2 !const
! 		! real*8 Err1
! 		! Err1 = 1.e-2
! 		if (.NOT. ( abs(xi(1))>Err1 .AND. abs(xi(2))>Err1 .AND. abs(xi(3))>Err1 ) ) then
! 			ax = 0.
! 			ay = 0.
! 			az = 0.

! 			if(.NOT. ( abs(xi(1))>Err1 ) ) then
! 				x1 = Err1
! 			end if
! 			if(.NOT. ( abs(xi(2))>Err1 ) ) then
! 				y1 = Err1
! 			end if
! 			if(.NOT. ( abs(xi(3))>Err1 ) ) then
! 				z1 = Err1
! 			end if
!         	get_acc_xyz(x1, y1, z1, accx, accy, accz)
! 			ax += accx
! 			ay += accy
! 			az += accz
			
! 			if(.NOT. ( abs(xi(1))>Err1 ) ) then
! 				x1 = -Err1
! 			end if
! 			if(.NOT. ( abs(xi(2))>Err1 ) ) then
! 				y1 = -Err1
! 			end if
! 			if(.NOT. ( abs(xi(3))>Err1 ) ) then
! 				z1 = -Err1
! 			end if
!         	get_acc_xyz(x1, y1, z1, accx, accy, accz)
! 			ax += accx
! 			ay += accy
! 			az += accz

! 			ai(1) = ax/2
! 			ai(2) = ay/2
! 			ai(3) = az/2
! 			! ai(1) = 1.
! 			! ai(2) = 1.
! 			! ai(3) = 1.
! 			adoti(1) = 0. !??
! 			adoti(2) = 0.
! 			adoti(3) = 0.
! 		else
! 			get_force_uncheck( xi, vi, ai, adoti )
! 		end if
! 		return
! 	end
! C######################################################


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


