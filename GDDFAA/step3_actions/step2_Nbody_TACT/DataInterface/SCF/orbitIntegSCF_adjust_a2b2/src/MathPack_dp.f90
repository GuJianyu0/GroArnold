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

!###########################################################
!!gjy note: code from CSDN
subroutine swap_real8_size2( a, b ) !gjy add
    real*8 :: a, b, t
    t = a
    a = b
    b = t
    return 
end subroutine

subroutine sort_array_1d( src, length, follower ) !gjy add: a better one is fortran: qsort()
    !!== To sort 1d array.
    !!== \param src     : target 1d array;
    !!== \param length  : length of \param src.
    !!== calling example:
    !!== call subr(src, size(src))
    implicit none
    integer :: length, i, j, min
    real*8 :: tmp
    real*8 :: src(length), follower(length) !length is size(src)

    do i=1,length-1
        min=i
        do j=i+1,length
            if (src(j) < src(min)) then
                min=j
            end if
        end do

        ! tmp = src(min)
        ! src(min) = src(i)
        ! src(i) = tmp
        call swap_real8_size2(src(i), src(min))
        call swap_real8_size2(follower(i), follower(min))
    end do
    return 
end subroutine

subroutine interpolate_3o_spline_1d( x, y, n, sx, f, f1, m ) !gjy add
    !!== 1-dim 3-order spline interpolation.
    !!== \param x   : x value of sample points;
    !!== \param y   : y value of sample points;
    !!== \param n   : count of sample points;
    !!== \param sx  : x value of target points;
    !!== \param f   : y value of target points;
    !!== \param f1  : dy/dx value of target points;
    !!== \param m   : count of target points.
    !!== calling example:
    !!== call subr( x, y, n, sx, f, f1, m )

    implicit none
    integer            :: i, j, k, m, n, n1, n2
    integer, parameter :: dp = 8
    Real(dp)           :: x(n), y(n), sx(m), f(m), f1(m)
    Real(dp)           :: s2(n), h(n-1), dy(n-1), s(n-1), e(n-2)
    Real(dp)           :: z, h1, h2, h3, h4
    n1 = n-1
    n2 = n-2

    if ( .NOT.( n>0 .AND. m>0 ) ) then
        print*, "Wrong length of sample or target points!"
        stop
    end if

    ! \param x, y has been changed by sort procedure
    call sort_array_1d(x, size(x), y)
    do k = 1, m
        if ( .NOT.( x(1)<sx(k) .AND. sx(k)<x(n) ) ) then
            print*, "x value of target points is out of range of sample points!"
            stop
        end if
    end do

    do i = 1, n1
        h(i)  = x(i+1) - x(i)
        dy(i) = ( y(i+1) - y(i) ) / h(i)
    end do
    
    s2(1) = 0.d0; s2(n) = 0.d0
    do i = 2, n1
        s2(i) = 6.d0 * ( dy(i) - dy(i-1) )
    end do
    
    z = 0.5d0 / ( h(1) + h(2) )
    s(1) = -h(2) * z
    e(1) = s2(2) * z
    do i = 2, n2
        k    = i - 1
        j    = i + 1
        z    = 1.d0 / ( 2.d0*( h(i)+h(j) ) + h(i)*s(k) )
        s(i) = -h(j) * z
        e(i) = ( s2(j)-h(i)*e(k) ) * z
    end do
    
    s2(n1) = e(n2)
    do i = n2, 2, -1
        k     = i - 1
        s2(i) = s(k)*s2(i+1) + e(k)
    end do
    
    do i = 1, n1
        s(i) = ( s2(i+1) - s2(i) ) / h(i)
    end do
    
    i = 2
    k = 1
    do j = 1, m
        do 
            if ( sx(j) > x(i) ) then
                k = i
                i = i + 1
            else
                exit
            end if
        end do
        h1    = sx(j) - x(k)
        h2    = sx(j) - x(i)
        h3    = h1 * h2
        h4    = s2(k) + h1*s(k)
        z     = ( s2(i) + s2(k) + h4 ) / 6.d0
        f(j)  = y(k) + h1*dy(k) + h3*Z
        f1(j) = dy(k) + z*( h1+h2 ) + h3 * s(k) / 6.d0
    end do
end subroutine
!###########################################################

#endif