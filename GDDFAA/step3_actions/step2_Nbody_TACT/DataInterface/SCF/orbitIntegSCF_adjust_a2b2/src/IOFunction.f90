!******************************************************************************
!    File   : IOFunction.f90
!
!    Func.  : write useful data to file/screen
!           : MPI version, cannot be used in
!           : other place
!------------------------------------------------------------------------------
!    Nobody is just for fun...

!    Begin at 2:00 p.m. Sep. 21th, 2010
!    Main part finish at 8:32 p.m. Sep. 21th, 2010
!******************************************************************************

#ifndef IOFunctionF90
#define IOFunctionF90

!######################################################
!/* To remove possible incorrect count of lines (not too much) */ !gjy add
subroutine CleanOrbitFile(Fname)
  implicit none
  character(*), intent(in) :: Fname

  integer :: i, ios, nfloat, nrec
  real*8  :: tmp
  real*8, allocatable :: buf(:)

  ! First pass: count how many reals are in the file (stream-wise).
  nfloat = 0
  open(unit=52, file=trim(Fname), status='old', &
       form='formatted', action='read', iostat=ios)
  if (ios /= 0) then
     print *, '[CleanOrbitFile] Cannot open ', trim(Fname)
     return
  end if

  do
     read(52, *, iostat=ios) tmp
     if (ios /= 0) exit
     nfloat = nfloat + 1
  end do

  rewind(52)

  if (nfloat <= 0) then
     print *, '[CleanOrbitFile] No data in ', trim(Fname)
     close(52)
     return
  end if

  allocate(buf(nfloat))

  ! Second pass: actually read all floats.
  do i = 1, nfloat
     read(52, *, iostat=ios) buf(i)
     if (ios /= 0) then
        nfloat = i - 1
        exit
     end if
  end do
  close(52)

  ! Each orbit record is: t, x(3), v(3), pot -> 8 numbers.
  nrec = nfloat / 8
  if (nfloat - 8*nrec > 0) then
     print *, '[CleanOrbitFile] Warning: ', trim(Fname), &
              ' has ', nfloat, ' floats, truncating to ', 8*nrec
  end if

  ! Rewrite the file with only full 8-column records.
  open(unit=52, file=trim(Fname), status='replace', &
       form='formatted', action='write', iostat=ios)
  if (ios /= 0) then
     print *, '[CleanOrbitFile] Cannot reopen ', trim(Fname), ' for writing.'
     deallocate(buf)
     return
  end if

  do i = 1, nrec
     write(52,"(1pe14.6, 2x, 7(1pe14.6))") &
          buf(8*(i-1)+1), buf(8*(i-1)+2), buf(8*(i-1)+3), buf(8*(i-1)+4), &
          buf(8*(i-1)+5), buf(8*(i-1)+6), buf(8*(i-1)+7), buf(8*(i-1)+8)
  end do

  close(52)
  deallocate(buf)

  print *, '[CleanOrbitFile] Cleaned ', trim(Fname), ' records = ', nrec
end subroutine CleanOrbitFile
!######################################################

!######################################################
!/*       output orbit info          */
subroutine WriteToFile( t, x, v, n )
  implicit none
  character*50 Fname
  real*8 x(3), v(3), t, r, pot
  integer n, nctrl, Factor, Major, idx, ecode
  
  common /SyS/ ecode, Major, idx
  
  Factor = 1
  
  call vector_mod( x, r )
  if( r > 10.0 ) then
    nctrl = 128*Factor
  else if( r > 1.0 ) then
    nctrl = 64*Factor
  else
    nctrl = 32*Factor
  end if
  nctrl = 1
  if( mod(n,nctrl)==0 ) then
    call get_pot( x, pot ) !/* Find this in get_force.f90 */
    
!!!! Fname = 'XXXXXX-YYYY.Orbit.dat' !!!!
!    write(Fname,"(i6.6,'-',i4.4)") Major, idx
!    Fname(12:21)='.Orbit.dat'
!!!! Fname = 'XXXXXX.Orbit.dat' !!!!
!   Major = particle id
!    write(Fname,"(i6.6)") Major

    ![learn code] the default status of open() is "old", one can change it to "replace"
    ![learn code] if add rewind write when "new", it will write only one line
    ![learn code] it will occupy the file (15??), C/C++ cannot write

    !    Fname(7:16)='.Orbit.dat'
    ! Use a dedicated unit for orbit output to avoid collisions with
    ! other uses of unit 15 (Integrate.status, STOP logging, etc.).
    open(unit=51,file='Orbit.dat',status='unknown', &
         position='append',action='write',form='formatted')

    ! ! REWIND(15)
    ! !gjy add: rewind
    ! write(15,"(1pe14.6, 2x, 7(1pe14.6)) ") t, x(1:3), v(1:3), pot
    ! close(15)

    ! REWIND(51)
    !gjy add: rewind
    write(51,"(1pe14.6, 2x, 7(1pe14.6)) ") t, x(1:3), v(1:3), pot
    close(51)
  end if
  
end subroutine WriteToFile
!######################################################

!######################################################
!/*       output orbit info          */ !gjy add
subroutine WriteToFile_Fname( t, x, v, n, Fname )
  implicit none
  character*50 Fname
  real*8 x(3), v(3), t, r, pot
  integer n, nctrl, Factor, Major, idx, ecode
  
  common /SyS/ ecode, Major, idx
  
  Factor = 1
  
  call vector_mod( x, r )
  if( r > 10.0 ) then
    nctrl = 128*Factor
  else if( r > 1.0 ) then
    nctrl = 64*Factor
  else
    nctrl = 32*Factor
  end if
  nctrl = 1
  if( mod(n,nctrl)==0 ) then
    call get_pot( x, pot ) !/* Find this in get_force.f90 */
    
!!!! Fname = 'XXXXXX-YYYY.Orbit.dat' !!!!
!    write(Fname,"(i6.6,'-',i4.4)") Major, idx
!    Fname(12:21)='.Orbit.dat'
!!!! Fname = 'XXXXXX.Orbit.dat' !!!!
!   Major = particle id
!    write(Fname,"(i6.6)") Major

    !    Fname(7:16)='.Orbit.dat'
    ! Dedicated unit and trimmed filename; avoids clashes with unit 15
    ! (Integrate.status etc.) and with any old Orbit.dat usage.
    open(unit=51,file=trim(Fname),status='unknown', &
         position='append',action='write',form='formatted')
    
    ! ! REWIND(15)
    ! !gjy add: rewind
    ! write(15,"(1pe14.6, 2x, 7(1pe14.6)) ") t, x(1:3), v(1:3), pot
    ! close(15)

    ! REWIND(51)
    !gjy add: rewind
    write(51,"(1pe14.6, 2x, 7(1pe14.6)) ") t, x(1:3), v(1:3), pot
    close(51)
  end if
  
end subroutine WriteToFile_Fname
!######################################################

!######################################################
!/*       release open file          */
subroutine WriteToFile_release
  implicit none
  ! open(unit=15,file='Orbit.dat',status='replace')
  ! rewind(15)
  ! close(15)

  open(unit=51,file='Orbit.dat',status='replace', &
       action='write',form='formatted')
  close(51)
end subroutine WriteToFile_release
!######################################################

!######################################################
!/*       release open file          */ !gjy add
subroutine WriteToFile_release_Fname(Fname)
  implicit none
  character*(*) Fname
  ! integer :: idx
  ! character*8 idxc
  ! character*50 filename
  ! write(idxc, '(I8)') idx
  ! idxc = ADJUSTL(idxc)  ! 移除左边的空格
  ! filename = TRIM(ADJUSTL('Orbit_'))//TRIM(ADJUSTL(idxc))//TRIM(ADJUSTL('.dat'))
  ! filename = ADJUSTL(filename)
  ! print*, filename

  ! ! open(unit=15,file=filename,status='replace')
  ! character*50 Fname
  ! open(unit=15,file=Fname,status='replace')
  ! ! open(unit=15,file='Orbit.dat',status='replace')
  ! rewind(15)
  ! close(15)
  
  ! open(unit=51,file=filename,status='replace')
  open(unit=51,file=trim(Fname),status='replace', &
       action='write',form='formatted')
  close(51)
end subroutine WriteToFile_release_Fname
!######################################################

!######################################################
!!! Only root process can call this subroutine !!!
!######################################################
subroutine ReadData( x, y, z, vx, vy, vz, Pidx, OrbIdx, nsmpl )
  implicit none
  
  character*50 buff, Fname
  real*8 x(0:1024000), y(0:1024000), z(0:1024000)
  real*8 vx(0:1024000), vy(0:1024000), vz(0:1024000)
  real*8 t, E, Jz, rtmp
  integer Di, i, itmp, nsmpl
  integer Pidx(0:1024000), OrbIdx(0:1024000)

  write(*,*) "读取 data.mpi.inp ……"
  open(unit=12,file='data.mpi.inp',status='old',err=100)
  i=0
  do
    read(12,*,end=12) Pidx(i), OrbIdx(i), x(i), y(i), z(i), vx(i), vy(i), vz(i)
    i = i + 1
  end do
12 close(12)
  nsmpl = i

!  write(*,*) "读取 gadget.out ……"
!  open(unit=12,file='gadget.out',status='old',err=101)
!  read(12,*) itmp
!  read(12,*) nsmpl
!  read(12,*) rtmp
!
!  do i=0,nsmpl-1
!    read(12,*) x(i), y(i), z(i), vx(i), vy(i), vz(i), rtmp
!    Pidx(i)=i
!    OrbIdx(i)=0
!  end do
!  close(12)
  
  if( nsmpl > 1024001 ) then
    write(*,"('Error:: sample size is larger than _1024001_ !')")
    write(*,"('Error:: nsmpl = ', i5)") nsmpl
    stop
  end if
  return

100 write(*,"('Cannot find data.mpi.inp. STOP')")
  stop
101 write(*,"('Cannot find gadget.out. STOP')")
  stop
end subroutine ReadData
!######################################################

!######################################################
subroutine SurfaceOfSec( t, x, v, n_sos )
  implicit none
  real*8 t, x(3), v(3), xold(3), vold(3)
  real*8 e_tot, e_pot, v_R, v_T, r
  integer idx, n_sos

  common /dat/ xold, vold
  
  if(xold(3)*x(3)<=0.0) then
      r   = sqrt( x(1)*x(1) + x(2)*x(2) )
      v_R = ( x(1)*v(1) + x(2)*v(2) ) / r
      if(v(1)*v(1) + v(2)*v(2) - v_R*v_R>=0.0) then
        v_T = sqrt(v(1)*v(1) + v(2)*v(2) - v_R*v_R)
      else
        v_T = 0.0
      end if
!      call get_pot( x, e_pot )
!      e_tot = 0.5*( v(1)*v(1) + v(2)*v(2) + v(3)*v(3) ) + e_pot
!      if( E < Emin .or. E > Emax ) cycle
      if( v(3) > 0.0 ) call WriteSoS( r, v_R, v_T, 'p' )
      if( v(3) < 0.0 ) call WriteSoS( r, v_R, v_T, 'n' )
      n_sos = n_sos + 1
  end if

end subroutine SurfaceOfSec
!######################################################

!################################################
subroutine CrossPoint( t,x,n_cp )
  implicit none
  real*8 t, x(3), xold(3), vold(3), xmid(3)
  real*8 e_tot, e_pot, v_R, v_T, r
  integer n_sos, n_cp(3)
  integer idx, Major, ecode
  character*50 Fname, ctmp

  common /dat/ xold, vold
  common /SyS/ ecode, Major, idx

  !!!! Fname = 'XXXXXX-YZ.dat' !!!!
  ! 穿过 YZ 平面 (x=0)
  if( xold(1)*x(1) <= 0.0 ) then
    xmid = 0.5*(xold+x)
    write(Fname,"(i6.6)") Major
    Fname(7:13)= '-YZ.dat'
    open(unit=21,file=Fname(1:14),position='append')
    write(21,*) t, xmid(2:3)
    close(21)
    n_cp(1)=n_cp(1)+1
  end if

  ! 穿过 XZ 平面 (y=0)
  if( xold(2)*x(2) <= 0.0 ) then
    xmid = 0.5*(xold+x)
    write(Fname,"(i6.6)") Major
    Fname(7:13)= '-XZ.dat'
    open(unit=21,file=Fname(1:14),position='append')
    write(21,*) t, xmid(1), xmid(3)
    close(21)
    n_cp(2)=n_cp(2)+1
  end if

  ! 穿过 XY 平面 (z=0)
  if( xold(3)*x(3) <= 0.0 ) then
    xmid = 0.5*(xold+x)
    write(Fname,"(i6.6)") Major
    Fname(7:13)= '-XY.dat'
    open(unit=21,file=Fname(1:14),position='append')
    write(21,*) t, xmid(1:2)
    close(21)
    n_cp(3)=n_cp(3)+1
  end if

end subroutine CrossPoint
!################################################



!################################################
subroutine WriteSoS( r, v_R, v_T, c )
    implicit none
    real*8 r, v_R, v_T
    integer idx, Major, ecode
    character c
    character*50 Fname, ctmp
    
    common /SyS/ ecode, Major, idx
    
    if( c == 'p' ) ctmp(1:8)= '.SoS.pos'
    if( c == 'n' ) ctmp(1:8)= '.SoS.neg'
    
    !!!! Fname = 'XXXXXX-YYYY.SoS.xxx' !!!!
    !write(Fname,"(i6.6,'-',i4.4)") Major, idx
    !Fname(12:19)= ctmp(1:8)
    !open(unit=21,file=Fname(1:19),position='append')
    
    !!!! Fname = 'XXXXXX.SoS.xxx' !!!!
    write(Fname,"(i6.6)") Major
    Fname(7:14)= ctmp(1:8)
    open(unit=21,file=Fname(1:14),position='append')
    write(21,*) r, v_R, v_T
    close(21)

end subroutine WriteSoS
!################################################


!################################################
!################################################
!################################################
!###############################################################
subroutine CheckStatus( t, x, v, Flag, Jmod, J, Major, Pid, n_peri, theta, my_rank )
  implicit none
  
  character*80 Fname
  integer Flag, Pid, n_peri, Major, my_rank
  
  real*8 t, rtmp(19)
  real*8 x(3), v(3), rsb, Jmod, J(3)
  real*8 rt, theta
  
  rt = 1.00000000E-003
  
  if( t <= 100.0 ) then
    rsb = x(1)**2. + x(2)**2. + x(3)**2.
    rsb = sqrt(rsb)
    if( rsb <= rt ) then
      write(Fname,"('In-loss-wedge.',i2.2)") my_rank
      open(unit=21,file=Fname(1:16),position='append')
      write(21,"(i3.3,1x,i5.5,1x,5(1pe12.3))") Major, Pid, Jmod, J(1:3), theta  !!! Initial value, not current one!!!
      close(21)
      
!      rtmp = 0.0
!      rtmp(8:10) = x(1:3)
!      rtmp(14:16) = v(1:3)
!      open(unit=21,file='accr-data.dat',position='append')
!      write(21,"(i5.5,1x,3(i2),2x,19(1pe12.3))") Pid,0,0,0, rtmp(5:19)
!      close(21)
      
      write(Fname,"('Orbit.parm.',i2.2)") my_rank
      open(unit=21,file=Fname(1:13),position='append')
      write(21,"(i3.3,1x,i5.5,2x,i3.3)") Major, Pid, n_peri
      close(21)
      
      Flag = 1
    end if
  else
!    open(unit=21,file='zzzOut-of-loss-wedge.dat',position='append')
!    write(21,"(i5.5,5(1pe12.3))") Pid, Jmod, J(1:3), theta  !!! Initial value, not current one!!!
!    close(21)
    Flag = -1
  end if
  
  
end subroutine CheckStatus
!###############################################################

!###############################################################
subroutine get_N_apo( n_apo, n_peri, xseq, x )
  implicit none
  integer n_apo, n_peri
  real*8 xseq(3,2), x(3), rtmp(3)
  real*8 d1, d2, d3
  
  rtmp(1:3) = xseq(1:3,1)
  call vector_mod( rtmp, d1 )
  
  rtmp(1:3) = xseq(1:3,2)
  call vector_mod( rtmp, d2 )
  
  rtmp(1:3) = x(1:3)
  call vector_mod( rtmp, d3 )
  
  if( d2 > d1 .and. d2 > d3 ) n_apo = n_apo + 1
  if( d2 < d1 .and. d2 < d3 ) n_peri = n_peri + 1
  
  xseq(1:3,1) = xseq(1:3,2)
  xseq(1:3,2) = x(1:3)
  
end subroutine get_N_apo
!###############################################################

#endif