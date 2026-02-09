
#ifndef read_dataF90
#define read_dataF90

subroutine read_data(x,y,z,mass,n)
  implicit none
  real*8 x(2048000),y(2048000),z(2048000),mass(2048000),tmp
  integer i,n,ii

  !gadget.out 由 halo+vel+pot.out改名而来
  open(unit=10,file="data.inp")
  REWIND(10)
  !gjy add: rewind
  read(10,*) ii
  read(10,*) n
  read(10,*) tmp

  do i=1,n
    read(10,*) mass(i),x(i),y(i),z(i),tmp,tmp,tmp,tmp
  end do

  close(10)

end subroutine read_data

#endif