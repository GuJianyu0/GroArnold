
program read_snapshot

   !------------------------------------------------------------------
   ! This Fortran 90 routine shows how a snapshot of GADGET
   ! which is distributed into several files may be read in.
   ! In this example, coodinates of particles type 1 are read.
   !
   ! Each dump consists of several files, each with a variable 
   ! number of particles. However, these files all have the same format. 
   ! The sequence of information in the files is as follows:
   !
   ! Header
   ! Positions
   ! Velocities
   ! Particle ID's
   !  
   ! Masses (for particle types with vanishing mass entry in the header)
   ! Internal Energy (Gas)
   ! Density (Gas)
   !
   !-------------------------------------------------------------------


   implicit none
   
   integer, parameter :: SNAPSHOT = 0       ! number of dump
   integer, parameter :: FILES = 8          ! number of files per snapshot

   character*200, parameter :: path='/afs/mpa/data/volker/lcdm_cr/'



   character*200 filename, snumber, fnumber

   integer*4 npart(0:5), nall(0:5)
   real*8    massarr(0:5)
   real*8    a
   real*8    redshift
   integer*4 unused(34)
   integer*4 fn,i,nstart,flag_sfr,flag_feedback
   integer*4 N,Ntot


   real*4,allocatable    :: pos(:,:),vel(:,:)
   integer*4,allocatable :: id(:)

   real*4,allocatable    :: PartPos(:,:)



   
   ! Ok, first we just see how many particles there are 
   ! of each type

   write(snumber,'(I3)') SNAPSHOT
   write(fnumber,'(I3)') 0
   do i=1,3 
      if (snumber(i:i) .eq. ' ') then 
          snumber(i:i)='0'
      end if
   end do

   filename= path(1:len_trim(path)) // '/snapshot_' // snumber(1:len_trim(snumber)) // '.' // fnumber(verify(fnumber,' '):3)

   print *,'opening...  '//filename(1:len_trim(filename))

   ! now, read in the header

   open (1, file=filename, form='unformatted')
   read (1) npart, massarr ,a, redshift, flag_sfr,flag_feedback, nall, unused
   close (1)


   print *,'Redshift=',redshift

   do i=1,6
      print *,'Type=',i,'    Particles=', nall(i)
   end do
   

   ! Now we just read in the coordinates, and only keep those of type=1
   ! The array PartPos(1:3,1:Ntot) will contain all these particles


   Ntot= nall(1)
   allocate(PartPos(1:3, 1:Ntot))

   nstart=0
   do fn=0, FILES-1 
      write(snumber,'(I3)') SNAPSHOT
      write(fnumber,'(I3)') fn
      do i=1,3 
         if (snumber(i:i) .eq. ' ') then 
            snumber(i:i)='0'
         end if
      end do
      filename= path(1:len_trim(path)) // '/snapshot_' // snumber(1:len_trim(snumber)) // '.' // fnumber(verify(fnumber,' '):3)
      print *,'reading...  '//filename(1:len_trim(filename))


      ! now, read in the file

      open (1, file=filename, form='unformatted')
      read (1) npart, massarr, a, redshift, flag_sfr,flag_feedback, nall, unused
      
      N=sum(npart)

      allocate(pos(1:3,1:N))

      read (1) pos
                         ! Note: If only the positions are needed,
                        ! only those have to be read in !
      close (1)


      PartPos(1:3,1+nstart:nstart+npart(1))=pos(1:3, 1 + npart(0)):sum(npart(0:1)))
      nstart=nstart+npart(1)

      deallocate(pos)
   end do

   print *,'Done with reading.'
 

   ! At this point, the coordinates of all particles of type 1 will
   ! be stored in the array PartPos(,)

end program




