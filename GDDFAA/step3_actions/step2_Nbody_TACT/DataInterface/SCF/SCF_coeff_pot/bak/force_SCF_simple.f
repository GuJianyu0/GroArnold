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
	subroutine get_pot( xi, yi, zi, poti )
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
