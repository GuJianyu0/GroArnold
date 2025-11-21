!###########################################################
!   File   : MathPack_dp.f90
!   Func.  : Some useful subroutines
!
!   Contents:: vector_cross,  vector_mod, InnerAngle,
!              vector_dot, my_randnum
!
!###########################################################

#ifndef MathPack_dpF90
#define MathPack_dpF90

!#######################################################################################
subroutine vector_cross( a, b, c, cmod)
    implicit none
    
    real*8 a(3), b(3), c(3), cmod
    
    c=0.0
    cmod=0.0
    
    c(1) = a(2)*b(3) - a(3)*b(2)
    c(2) = a(3)*b(1) - a(1)*b(3)
    c(3) = a(1)*b(2) - a(2)*b(1)
    
    cmod = sqrt( c(1)**2. + c(2)**2. + c(3)**2.)
    
    return
end subroutine vector_cross
!#######################################################################################

!#######################################################################################
subroutine vector_dot( a, b, c)
    implicit none
    
    real*8 a(3), b(3), c
    
    c = a(1)*b(1) + a(2)*b(2) + a(3)*b(3)
    
end subroutine vector_dot
!#######################################################################################

!#######################################################################################
subroutine vector_mod( c, cmod)
    implicit none
    real*8 c(3), cmod
    cmod = sqrt( c(1)**2. + c(2)**2. + c(3)**2.)
    return
end subroutine vector_mod
!#######################################################################################

!#######################################################################################
subroutine InnerAngle( a, b, theta)
    implicit none
    
    real*8 a(3), b(3), theta
    real*8 ab, amod, bmod
    
    ab = a(1)*b(1) + a(2)*b(2) + a(3)*b(3)
    amod = sqrt( a(1)*a(1) + a(2)*a(2) + a(3)*a(3) )
    bmod = sqrt( b(1)*b(1) + b(2)*b(2) + b(3)*b(3) )
    
    theta = acos(ab/(amod*bmod))
    
end subroutine InnerAngle
!#######################################################################################

!###########################################################

subroutine my_randnum( z, SEED )  !Schrage method
    implicit none
    real*8 z
    integer z_tmp,SEED,a,m,q,r
    
    a = 16807
    m = 2147483647
    
    q = m / a
    r = mod( m , a )
    
    z_tmp = a * mod( SEED, q) - r * ( SEED / q )
    
    if( z_tmp >= 0 ) then
	z = real( z_tmp ) / real(m)
	SEED = z_tmp
	return
    else
	z = real( z_tmp + m ) / real(m)
	SEED = z_tmp + m
	return
    end if

end subroutine my_randnum

!###########################################################

#endif