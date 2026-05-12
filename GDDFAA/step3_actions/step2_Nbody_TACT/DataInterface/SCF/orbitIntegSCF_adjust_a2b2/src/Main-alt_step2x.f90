!******************************************************************************
!    File   : Hermit-4th.f90
!
!    Func.  : N-body simulation program...serial running
!           : Utilize the 4th Hermite Integration scheme
!           :
!           : Integrate the orbit of ONE star in given pot
!------------------------------------------------------------------------------
!    Nobody is just for fun...

!    Begin at 2:00 p.m. Sep. 21th, 2010
!    Main part finish at 8:32 p.m. Sep. 21th, 2010
!******************************************************************************
!###############################################################

! include 'IOFunction.f90'
! include 'MathPack_dp.f90'
! include 'calc_dt.f90'
! include 'get_time.f90'



! !###############################################################
! !###############################################################
! !###############################################################
! program main
!   implicit none
!   using orbintegx
  
!   integer n_call
  
!   real*8 x(3),v(3),pot,vmod,vstep,dist
!   real*8 v1,v2,v3
!   real*8 dy1,dy2,dy3
!   real*8 dy,res
!   character*8 buff

!   real tt(2),t0,t1

! !+++++++++++++++++++++++++++++++++++++++++
! !   Variables for SCF
! !+++++++++++++++++++++++++++++++++++++++++
!   real*8 sinlist(0:4095), coslist(0:4095)
!   integer ptr, nloop, loop_cnt
!   real*8 xlocal(3), vlocal(3), t_end
!   real*8 mbh, gconst
!   logical zo, ze
! !+++++++++++++++++++++++++++++++++++++++++


! ! 读入初始的y坐标
!   if( iargc()==1 ) then
!     call getarg(1,buff)
!     read(buff,*) dist
!   else
!     dist = 1.0
!     print*,"使用缺省的初始y坐标:",dist
!   end if


! !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! ! SCF 部分的初始化
! !  call ReadData( x_N, y_N, z_N, vx_N, vy_N, vz_N, Pidx, OrbIdx, nsmpl )

!   write(*,*) 'Read SCF Coeff'
!   call ReadCoeff( sinlist, coslist, mbh, zo, ze, gconst )
!   call set_parameter( sinlist, coslist, mbh, zo, ze, gconst )
! !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! ! 确定大致的搜索范围
!   x = 0.0
!   v = 0.0
!   n_call = 0
!   t0 = etime(tt)

!   ! 轨道的初始位置在Y轴上
!   x(2) = dist
  
!   call get_pot(x,pot)
!   vmod = sqrt(abs(pot))
!   print*,"y =",x(2), "v_circ=",vmod
!   vstep = vmod / 20.0
!   v1 = vmod
!   ! 从y_i 出发，积分半个轨道后与Y轴相交于 y_f (<0)
!   ! dy = abs(-y_f - y_i)
!   call compute_dy(x,v,v1,n_call,dy1)

!   v2 = v1 - vstep
!   call compute_dy(x,v,v2,n_call,dy2)
  
!   do
!     v3 = v2-vstep
!     if( v3 < 0.0 ) then
!       print*,"未能找到搜索区间，退出。"
!       stop
!     end if

!     call compute_dy(x,v,v3,n_call,dy3)

!     if( dy1 > dy2 .and. dy2 < dy3 ) exit
!     v1=v2
!     v2=v3
!     dy1 = dy2
!     dy2 = dy3

!   end do
!   ! 注意按照升序排列
!   print*,"搜索区间：", v3, v2, v1
!   print*,"n_call=",n_call

!   ! 使用golden section search方法找出使 dy 最小化的 v
!   call GSS(x,v,   v3,v2,v1,   n_call,   dy,res)
!   print*,"V_x = ", res
!   print*,"|Delta y| =",dy
!   print*,"n_call=",n_call

!   print*,"输出该轨道信息至文件 Orbit.dat"
!   ! 输出该轨道信息至文件
!   ! 绕x轴旋转的轨道
!   !v(3)=res
!   ! 绕z轴旋转的轨道
!   v(1)=res

!   call Integrator(x,v,1,dy)
!   t1 = etime(tt)
!   write(*,"('总计用时:',f9.3, ' 秒')") t1-t0

! end program main
! !#####################################################################
! !#####################################################################



!!!! wrap by module
module orbintegx
CONTAINS
!###############################################################
!###############################################################
!###############################################################
subroutine main_for_orbit_integrating (dist, Fname_orbit)
  implicit none
  character*50 Fname_orbit !gjy add
  
  real*8, optional :: dist
  integer n_call
  
  real*8 x(3),v(3),pot,vmod,vstep
  real*8 v1,v2,v3
  real*8 dy1,dy2,dy3
  real*8 dy,res
  character*8 buff

  real tt(2),t0,t1

!+++++++++++++++++++++++++++++++++++++++++
!   Variables for SCF
!+++++++++++++++++++++++++++++++++++++++++
  real*8 sinlist(0:4095), coslist(0:4095)
  integer ptr, nloop, loop_cnt
  real*8 xlocal(3), vlocal(3), t_end
  real*8 mbh, gconst
  logical zo, ze
!+++++++++++++++++++++++++++++++++++++++++


! 读入初始的y坐标
  ! if( iargc()==1 ) then
  !   call getarg(1,buff)
  !   read(buff,*) dist
  ! else
  !   dist = 1.0
  !   print*,"使用缺省的初始y坐标:", dist
  ! end if
  if(.not. present(dist)) then !gjy add
    dist = 1.0
    print*,"使用缺省的初始y坐标:", dist
  end if


!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! SCF 部分的初始化
!  call ReadData( x_N, y_N, z_N, vx_N, vy_N, vz_N, Pidx, OrbIdx, nsmpl )

  write(*,*) 'Read SCF Coeff'
  call ReadCoeff( sinlist, coslist, mbh, zo, ze, gconst )
  call set_parameter( sinlist, coslist, mbh, zo, ze, gconst )
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! 确定大致的搜索范围
  x = 0.0
  v = 0.0
  n_call = 0
  t0 = etime(tt)

  ! 轨道的初始位置在Y轴上
  x(2) = dist
  
  call get_pot(x,pot)
  vmod = sqrt(abs(pot))
  print*,"y =",x(2), "v_circ=",vmod
  vstep = vmod / 40.0 !gjy comment: total time each; original
  ! vstep = vmod / 20.0
  v1 = vmod
  ! 从y_i 出发，积分半个轨道后与Y轴相交于 y_f (<0)
  ! dy = abs(-y_f - y_i)
  call compute_dy(x,v,v1,n_call,dy1)

  v2 = v1 - vstep
  call compute_dy(x,v,v2,n_call,dy2)
  
  do
    v3 = v2-vstep
    if( v3 < 0.0 ) then
      print*,"未能找到搜索区间，退出。"
      stop
    end if

    call compute_dy(x,v,v3,n_call,dy3)

    if( dy1 > dy2 .and. dy2 < dy3 ) exit
    v1=v2
    v2=v3
    dy1 = dy2
    dy2 = dy3

  end do
  ! 注意按照升序排列
  print*,"搜索区间：", v3, v2, v1
  print*,"n_call=",n_call

  ! 使用golden section search方法找出使 dy 最小化的 v
  call GSS(x,v,   v3,v2,v1,   n_call,   dy,res)
  print*,"V_x = ", res
  print*,"|Delta y| =",dy
  print*,"n_call=",n_call

  print*,"输出该轨道信息至文件 Orbit.dat"
  ! 输出该轨道信息至文件
  ! 绕x轴旋转的轨道
  !v(3)=res
  ! 绕z轴旋转的轨道
  v(1)=res

  ! call Integrator(x,v,1,dy)
  call Integrator(x,v,1,dy, Fname_orbit) !gjy change
  t1 = etime(tt)
  write(*,"('总计用时:',f9.3, ' 秒')") t1-t0

end subroutine main_for_orbit_integrating
!#####################################################################
!#####################################################################



!#####################################################################
subroutine GSS(pos,vel,ax,bx,cx,n_call,dy,res)
  implicit none
  integer n_call
  real*8 ax,bx,cx,dy,res
  real*8 pos(3),vel(3)
  real*8 f1,f2,x0,x1,x2,x3
  real*8 R,C
  real*8,parameter::tol=1.0e-7

  R=0.61803399
  C = 1.0-R

  x0 = ax
  x3 = cx
  if( abs(cx-bx) > abs(bx-ax) ) then
    x1 = bx
    x2 = bx+C*(cx-bx)
  else
    x2 = bx
    x1 = bx-C*(bx-ax)
  end if

  call compute_dy(pos,vel,x1,n_call,f1)
  call compute_dy(pos,vel,x2,n_call,f2)

  do while( abs(x3-x0) > tol*(abs(x1)+abs(x2)) )
    if(f2<f1) then
      x0=x1
      x1=x2
      x2=R*x1+C*x3
      f1=f2
      call compute_dy(pos,vel,x2,n_call,f2)
    else
      x3=x2
      x2=x1
      x1=R*x2+C*x0
      f2=f1
      call compute_dy(pos,vel,x2,n_call,f1)
    end if
  end do

  if(f1<f2) then
    res = x1
    dy  = f1
  else
    res = x2
    dy  = f2
  end if

end subroutine GSS
!#####################################################################

!#####################################################################
subroutine compute_dy(x,v,set_v,n_call,dy)
  implicit none
  real*8 x(3),v(3),dy,set_v
  integer n_call
  real tt(2),t0,t1
  character*50 Fname_orbit_notused !gjy add
  Fname_orbit_notused = 'Orbit_compute_dy.dat' !gjy add

  ! 绕x轴旋转的轨道
  !v(3)=set_v
  ! 绕z轴旋转的轨道
  v(1)=set_v
  
  t0 = etime(tt)
  ! call Integrator(x,v,0,dy)
  call Integrator(x,v,0,dy, Fname_orbit_notused) !gjy change
  t1 = etime(tt)
  n_call = n_call+1
  ! 以防万一
  dy = abs(dy)
  write(*,"('V_x = ',f9.3, ' |Delta y|= ', f14.8, 2x,'time = ', f11.3)") set_v, dy, t1-t0

end subroutine compute_dy
!#####################################################################


!#####################################################################
! subroutine Integrator( x0, v0, outflag, dy )
  subroutine Integrator( x0, v0, outflag, dy, Fname_orbit ) !gjy change
    implicit none
!			/* parameter define */
   integer outflag
   real,parameter::DTMINPOWER = -40.0
   real,parameter::DTMAXPOWER = -10.0
   real,parameter::DT_CORR = 8.0
   logical,parameter:: exact_on_plane = .FALSE.

!/*  e.g. dt_min = 2.0 ** DTMINPOWER */
!/* DTMINPOWER		*/
!/* -10		10^-3	*/
!/* -13		10^-4	*/
!/* -16		10^-5	*/
!/* -20		10^-6	*/
!/* -23		10^-7	*/
!/* -26		10^-8	*/
!/* -30		10^-9	*/
   integer,parameter:: KB = 1024


!/* variable define */

character*50 inputFile, Fname, Fout, ctmp
character*50 Fname_orbit !gjy add
integer nFile
!	/* for the "filename=inputFile(1:nFile-1)"   */
integer N_MAX,idxFile
integer i, idxBH, idxTemp, ncontr
integer n_step, nctrl
integer i_act, ipow, my_rank

real   timeFile
real*8 m,x(3),v(3),x0(3),v0(3),a(3),a0(3),adot(3),adot0(3),a2dot(3),a3dot(3)
real*8 dy,ymin,ymax
real*8 xp(3),vp(3),ap(3),apdot(3)
real*8 t0,t_next
real*8 pot,r, rmin, min_dt
real*8 a_mod, adot_mod

real*8 t_global,dt_min,dt_max,t_t0	!/* time variables */
real*8 t_end				!/* end time */
real*8 dt_i,DTPOWER
real*8 timeTable(38)

!/* timing */
real   tt(2),eltime
real   timer_0

!/* time for info output */
real*8 t_info,t_info_step, t_save, t_save_step

!/* time for JEVO output */
real*8 t_Jevo,t_Jevo_step

!/* for energy control...*/
real*8 e_tot,e_tot_0,de_tot
real*8 e_kin,e_pot,e_tot_beg,e_tot_end

!/* for angular moment control...*/
real*8 J(3), Jmod, Jold(3), Joldmod, DJ(3), DJmod

real*8 xold(3), vold(3), yi, yf

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!!! For E-J-Jz loss-wedge calc

integer Flag, n_peri, n_apo, n_sos
real*8 xseq(3,2), Jmod0, J0(3), r3d, theta
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

integer ecode, my_Major, my_OrbIdx

common /dat/ xold, vold
common /SyS/ ecode, my_Major, my_OrbIdx
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

x=x0
v=v0

xold = x
vold = v

yi=x(2)
yf=yi

ymin=1.0e10
ymax=-1.0

t0 = 0.0
t_next = dt_min

t_global = 0.0
dt_min = 2.0 ** DTMINPOWER
dt_max = 2.0 ** DTMAXPOWER
t_info_step = 1.0/4.0
t_Jevo_step = 1.0/4.0

t_save_step = 1.0e-4
t_save = t_save_step

! t_end = 10.0 ! Gadget unit: Gyr
t_end = 100.0 ! Gadget unit: Gyr

!	/* initializing... */

DTPOWER = -3.
do i=1,38		!/* DTMINPOWER <= DTPOWER <= DTMAXPOWER */
    timeTable(i) = 2.**DTPOWER
    DTPOWER = DTPOWER - 1.
end do

xseq(1:3,1) = x(1:3) ; xseq(1:3,2) = x(1:3)
r3d = x(1)**2.0 +  x(2)**2.0 + x(3)**2.0
r3d = sqrt(r3d)
theta = acos(x(3)/r3d)

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!!!! Fname = 'Process-XX.out'!!!!
!write(Fname,"('Process-',i2.2)") my_rank
!Fname(11:14) = '.out'
!Fout = Fname
!open(unit=15,file=Fout(1:14),position='append')

call energy_contr( x, v, e_kin, e_pot, e_tot)
e_tot_beg = e_tot

call vector_cross( x, v, Jold, Joldmod )
Jmod0 = Joldmod; J0 = Jold

!if( my_Major > 1000) stop
! Orbital energy filter
!if( e_tot < E_low .or. e_tot > E_high ) then
!  write(15,"(i6.6,'-',i4.4,' Skip |',$ )") my_Major, my_OrbIdx
!  write(15,"('E_tot = ',1pe11.3 )") e_tot
!  return
!end if

if( e_tot >= 0.0 ) then
  write(*,"('Initial condition has a positive energy!',1pe15.7)") e_tot
  write(*,"('++++++++++++++++++++++++++++++++++')")
  stop
end if

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ecode = 0
call get_force( x, v, a0, adot0 )
if( exact_on_plane ) then
  a0(3)=0.0
  adot0(3)=0.0
end if

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++ Determine the first time step +++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

a_mod = a0(1)**2.0 + a0(2)**2.0 +a0(3)**2.0 
adot_mod = adot0(1)**2.0 + adot0(2)**2.0 + adot0(3)**2.0

dt_i = 0.01*sqrt(a_mod/adot_mod)
ipow = log(dt_i)/log(2.0) - 1
dt_i = 2.0**real(ipow) / 32.0

if( dt_i < dt_min ) dt_i = dt_min

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

t_global = t_global + dt_i			!/* one time step forward... */
t_info = t_info_step
t_Jevo = t_Jevo_step

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

eltime = etime(tt)
timer_0 = eltime

n_step=0
rmin = 1.000D+02
min_dt = 1.000D+00

Flag = 0
n_peri = 0
n_apo = 0
n_sos = 0


open(unit=15,file='Integrate.status',position='append')
! REWIND(15)
!gjy add: rewind
write(15,"('--------------Integration Start-------------------')")
close(15)


! 循环结束的三个控制条件：
! t_end 最大演化时间，Gyr
! n_apo 粒子经过远心点的次数
! n_sos 有效 Surface of Section 记录数
do while( t_global <= t_end .and. n_apo < 20 )
!do while( n_step <= 30 )
!	/* PREDICTION for PARTICLE */

  t_t0 = t_global - t0

  !/* coordinate */
  xp = x + v * t_t0 + 0.5 * a0 * (t_t0**2.0) + adot0 * (t_t0**3.0) / 6.0

  !/* velocity */
  vp = v + a0 * t_t0 + 0.5 * adot0 * (t_t0**2.0)

  !/* calc force  */
  ecode=0
  call get_force( xp, vp, ap, apdot )  !/* calculate force felt by the active particles, using predicted coor. */
  if( exact_on_plane) then
    ap(3)=0.0
    apdot(3)=0.0
  end if
  call check_NAN(vp,ecode)
  call check_NAN(ap,ecode)
  call check_NAN(apdot,ecode)
  if( ecode == 99 ) then
    open(unit=16,file="NaN-record",position='append')
    write(16,*) "NaN occurs:", my_Major
    write(16,*) "xp ", xp(1:3)
    write(16,*) "vp ", vp(1:3)
    write(16,*) "ap" , ap(1:3)
    write(16,*) "apdot ", apdot(1:3)
    close(16)
    stop
  end if

  !/* for all the active particles, correct their coor. & vel. etc... */

  t_t0 = t_global - t0
  a2dot = -6.0 * ( a0 - ap ) / (t_t0**2.0) - 2. * (2. * adot0 + apdot ) / t_t0
  a3dot = 12.  * ( a0 - ap ) / (t_t0**3.0) + 6. * (     adot0 + apdot ) / (t_t0**2.0)

!		/* CORRECTION */
!/* coordinate */
  x = xp + a2dot * (t_t0**4.) / 24. + a3dot * (t_t0**5.) / 120.
!/* velocity */
  v = vp + a2dot * (t_t0**3.) / 6.  + a3dot * (t_t0**4.) / 24.
!/* get next integration time */

  call calc_dt( a0, adot0, a2dot, a3dot, dt_i )
  call reduce_dt( dt_i, t_t0, t_global, dt_min, dt_max )
    
  t0 = t_global     !/* synchronize particle's local time to global time */

  if( dt_i < min_dt ) min_dt = dt_i
  call vector_mod(x,r)
  if( r < rmin ) rmin = r

!/* this timestep's calc. is over, move on... */

  t_global = t_global + dt_i
!*************************************


!+++++++++++++++++++++++++++++++++++!/* next line maybe very important... */
!!!  ecode=0
!!!  call get_force( x, v, a0, adot0 )  !/* calc a , adot , pot for active particles, using corrected coor. */
!!!  if( ecode == 1 ) then
!!!    open(unit=15,file=Fout(1:14),position='append')
!!!    write(15,"('Reach boundary!')")
!!!    close(15)
!!!    return
!!!  end if
   
   a0 = ap; adot0 = apdot
   
!+++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++
! 输出监控信息
  if( t_global >= t_info ) then	!/* useful info output */
    ! 检测是否要强制停止，如果存在名为 STOP的文件则停止模拟
    open(unit=99,file='STOP',status='old',err=999)
    write(15,*) "Manually STOP."
    flush(15)
    stop
999 continue
    eltime = etime(tt)
    eltime = eltime - timer_0
    
    t_info = t_info + t_info_step


    e_tot_0 = e_tot
    call energy_contr( x, v, e_kin, e_pot, e_tot) !/* Find this in get_force.f90 */
    de_tot = e_tot - e_tot_0
    e_tot_end = e_tot
    
    open(unit=15,file='Integrate.status',position='append')
    write(15,"('t = ',1pe11.3,2x, 'n_step = ',i9,2x, 'min_dt = ', 1pe11.3,$)") t_global, n_step, min_dt
    write(15,"(2x,'r_min = ',1pe11.3, 2x, 'n_apo = ', i3, 2x, 'e_tot =',1pe24.16)") rmin, n_apo, e_tot
    close(15)

    rmin = 1.000D+02 ;   min_dt = 1.000D+02

  end if !/* t_global > t_info */
! 输出监控信息，结束


! 输出轨道信息
  if( t_global >= t_save .and. outflag==1 ) then
    ! call WriteToFile( t_global, x, v, n_step ) !gjy comment: open() Orbit.dat
    call WriteToFile_Fname( t_global, x, v, n_step, Fname_orbit ) !gjy change
    t_save = t_save + t_save_step
  end if
!  if( abs(x(3)) < 0.01 ) call SurfaceOfSec( t_global, x, v, n_sos )

!+++++++++++++++++++++++++++++++++++++++++++++++++++++
! 如果循环的wallclock time太长，可能出现了问题，终止该粒子的积分
  eltime = etime(tt)
  if( eltime - timer_0 > 600 ) then
    ! open(unit=15,file=Fout(1:14),position='append') !gjy change: comment out; Fout has been comment out before
    ! write(15,"(i6.6,'-',i4.4,2x, 'Error occurs in Integration.')") my_Major, my_OrbIdx
    ! close(15)
    print*,"eltime - timer_0 > 600. Now exit this function." !gjy add
    exit
  end if
!+++++++++++++++++++++++++++++++++++++++++++++++++++++

  call get_N_apo( n_apo, n_peri, xseq, x )
!  if( n_apo > 100 ) t_global = 10000.0
!  call CheckStatus( t_global, x, v, Flag, Jmod0, J0, my_Major, my_OrbIdx, n_peri, theta, my_rank )
!  if( Flag /= 0 ) exit

  n_step = n_step + 1

! 绕z轴旋转的轨道，在XY平面，记录多个整周期中在y轴距离的最大、最小值
! 在搜索vz的过程中，不输出到文件
  if( x(1)*xold(1)<0.0 .and. x(2)>0.0 .and. outflag==0 ) then ! 穿越Y轴
    yf = 0.5*(x(2)+xold(2))
    if( yf < ymin) ymin=yf
    if( yf > ymax) ymax=yf
  end if

  xold = x
  vold = v

end do	!/*do while( t_global < t_end ) */

dy = ymax-ymin

!  eltime = etime(tt)
!  eltime = eltime - timer_0
  
!  open(unit=15,file=Fout(1:14),position='append')
!  write(15,"(/, i5.5, '-', i4.4, 2x, 'Finished. '$)") My_Major, My_OrbIdx
!  write(15,"('n_apo = ', i4, ' n_sos = ',i4, ' status =', i3)") n_apo, n_sos, Flag
!  write(15,"('Wallclock time = ', f11.3, ' sec')") eltime
!  write(15,"(/'N_step= ',i10,' Speed= ', f11.3,' steps/sec'/)") n_step, real(n_step)/eltime
!  write(15,*) e_tot_beg, e_tot_end
!  write(15,"(i6.6,'-',i4.4,' | ',$)") my_Major, my_OrbIdx
!  write(15,"('E_diff/E_beg = ',1pe11.3)") (e_tot_end - e_tot_beg)/e_tot_beg
!  write(15,"('++++++++++++++++++++++++++++++++++')")
!  close(15)

end subroutine Integrator
!#####################################################################


!#####################################################################
subroutine check_NAN(x,ecode)
  use,intrinsic :: IEEE_ARITHMETIC

  implicit none
  real*8 x(3)
  character*8 cs
  integer i,ecode
  ecode=0
  do i=1,3
    if( IEEE_IS_NaN(x(i)) ) then
       !write(15,*) 'NaN occurs,', cs,x(1:3)
       ecode=99
       return
    end if
  end do

end subroutine check_NAN
!#####################################################################
end module orbintegx
!!!!end module
