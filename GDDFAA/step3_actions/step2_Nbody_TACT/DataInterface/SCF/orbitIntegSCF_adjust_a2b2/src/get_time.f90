subroutine get_time( time, E )
  implicit none
  
  real*8 time, E
  
  time = 1
  !!!return
  
  if( E > -1.0 ) then
     time = 300
  else if( E > -1.5 ) then
     time = 100
  else if( E > -2.0 ) then
     time = 80
  else if( E > -2.5 ) then
     time = 50
  else if( E > -3.0 ) then
     time = 30
  else
     time = 10
  end if

end subroutine get_time