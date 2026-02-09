!******************************************************************************
!    File   : calc_dt.f90
!
!    Func.  : N-body simulation program...serial running
!           : Utilize the 4th Hermite Integration scheme
!    Ver.   : 1.0.1 laohu
!------------------------------------------------------------------------------
!    Nobody is just for fun...

!    Begin at 2:00 p.m. Sep. 21th, 2010
!    Main part finish at 8:32 p.m. Sep. 21th, 2010
!******************************************************************************
!*******************************************************************************

#ifndef calc_dtF90
#define calc_dtF90

subroutine calc_dt( a0, adot0, a2dot, a3dot, dt_i )
	implicit none

	real*8 dt_i,timeTable(38)
	real*8 a0(3), adot0(3), a2dot(3), a3dot(3)
	real*8 a_mod,adot_mod,a2dot_mod,a3dot_mod
	
	real,parameter:: eta = 0.01
	
	a_mod = sqrt( a0(1)**2. + a0(2)**2. + a0(3)**2.)
	adot_mod  = sqrt( adot0(1)**2. + adot0(2)**2. + adot0(3)**2.)
	a2dot_mod = sqrt( a2dot(1)**2. + a2dot(2)**2. + a2dot(3)**2.)
	a3dot_mod = sqrt( a3dot(1)**2. + a3dot(2)**2. + a3dot(3)**2.)
	
	dt_i = eta * (( a_mod * a2dot_mod )+ adot_mod * adot_mod )
	dt_i = dt_i / ((adot_mod * a3dot_mod) + a2dot_mod * a2dot_mod )
	dt_i = sqrt( dt_i ) / 8.0
	
	! 强行限制 dt 的下限值
	do while(dt_i < 1.0e-7) !gjy note: original
          dt_i = dt_i*2.0D0
        end do
	return

end subroutine calc_dt
!*******************************************************************************

subroutine reduce_dt( dt_new, dt_old, min_t, dt_min, dt_max )
	implicit none
	real*8 dt_new, dt_old, min_t, dt_min, dt_max, dt_tmp
	real*8 getIndex
	integer n, dcnt(3), p
	
	dt_tmp = dt_old
	
	!print*, dt_new, 2.0*dt_tmp, min_t, int(min_t/dt_tmp), mod(min_t,2.0*dt_tmp)
	if( dt_new < dt_min ) then
	  dt_tmp = dt_min
	  dcnt(1) = dcnt(1)+1
	end if
	
	if( dt_new > dt_min .and. dt_new < dt_old ) then
	  p = log(dt_new)/log(2.000D+00)-1.0000D+00
	  dt_tmp = (2.0000D+00)**(real(p))
	  dcnt(2) = dcnt(2)+1
	end if
	
	!if( dt_new > dt_old .and. dt_old < dt_old*2.0 )  dcnt(2) = dcnt(2)+1
	
	if( dt_new > 2.00000D+00*dt_old .and. mod(min_t,2.00D+00*dt_old)==0.0 .and. 2.00D+00*dt_old<=dt_max) then
	  dt_tmp = dt_old*2.0
	  dcnt(3) = dcnt(3) + 1
	end if
	
	!print*, dt_new, 2.0*dt_tmp, min_t, int(min_t/dt_tmp), mod(min_t,2.0*dt_tmp)
	!print*, dt_new, getIndex(dt_new), 2.0*dt_tmp, getIndex(2.0*dt_tmp), min_t, getIndex(min_t), mod(min_t,2.0*dt_tmp)
	
	dt_new = dt_tmp
	return
end subroutine reduce_dt
!*******************************************************************************


!*************************************************
real*8 function getIndex( x )
  implicit none
  real*8 x
  getIndex = log(x)/log(2.0)

end function getIndex
!*************************************************

#endif