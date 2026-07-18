      SUBROUTINE UEXTERNALDB(LOP,LRESTART,TIME,DTIME,KSTEP,KINC)
      INCLUDE 'ABA_PARAM.INC'
      DIMENSION TIME(2)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
      REAL*8 PE1(310376)                               ! Size of PEMatrix matrix
      common /PE1matrix/ PE1                         ! Consider PEMatrix as a common 
      REAL*8 PE2(310376)                               ! Size of PEMatrix matrix
      common /PE2matrix/ PE2                         ! Consider PEMatrix as a common variable
      REAL*8 PE3(310376)                               ! Size of PEMatrix matrix
      common /PE3matrix/ PE3                         ! Consider PEMatrix as a common variable
      REAL*8 PE12(310376)                               ! Size of PEMatrix matrix
      common /PE12matrix/ PE12                         ! Consider PEMatrix as a common variable
      REAL*8 PE13(310376)                               ! Size of PEMatrix matrix
      common /PE13matrix/ PE13                         ! Consider PEMatrix as a common variable
      REAL*8 PE23(310376)                               ! Size of PEMatrix matrix
      common /PE23matrix/ PE23                         ! Consider PEMatrix as a common variable
	  REAL*8 ELEMENTNR(310376)
	  common /ELEMENTNRmatrix/ ELEMENTNR
	  REAL*8 IPNR(310376)
	  common /IPNRmatrix/ IPNR
	  IF (LOP==0) THEN                                ! Read the txt file only only once
          OPEN(UNIT=1, FILE="/cluster/scratch/mpatrik/UEXPAN/Eigenstrains_Batch1_11.txt") 
              READ(1,*) PE1
          CLOSE (1)
          OPEN(UNIT=2, FILE="/cluster/scratch/mpatrik/UEXPAN/Eigenstrains_Batch1_22.txt") 
              READ(2,*) PE2
          CLOSE (2)
          OPEN(UNIT=3, FILE="/cluster/scratch/mpatrik/UEXPAN/Eigenstrains_Batch1_33.txt") 
              READ(3,*) PE3
          CLOSE (3)
          OPEN(UNIT=4, FILE="/cluster/scratch/mpatrik/UEXPAN/Eigenstrains_Batch1_12.txt") 
              READ(4,*) PE12
          CLOSE (4)
          OPEN(UNIT=5, FILE="/cluster/scratch/mpatrik/UEXPAN/Eigenstrains_Batch1_13.txt") 
              READ(5,*) PE13
          CLOSE (5)
          OPEN(UNIT=9, FILE="/cluster/scratch/mpatrik/UEXPAN/Eigenstrains_Batch1_23.txt") 
              READ(9,*) PE23
          CLOSE (9)
		  OPEN(UNIT=11, FILE="/cluster/scratch/mpatrik/UEXPAN/CL24_Element.txt") 
              READ(11,*) ELEMENTNR
          CLOSE (11)
		  OPEN(UNIT=12, FILE="/cluster/scratch/mpatrik/UEXPAN/CL24_IP.txt") 
              READ(12,*) IPNR
          CLOSE (12)
      END IF
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      RETURN
      END
	  
	  SUBROUTINE USDFLD(FIELD,STATEV,PNEWDT,DIRECT,T,CELENT,
     + TIME,DTIME,CMNAME,ORNAME,NFIELD,NSTATV,NOEL,NPT,LAYER,
     + KSPT,KSTEP,KINC,NDI,NSHR,COORD,JMAC,JMATYP,MATLAYO,LACCFLA)
      INCLUDE 'ABA_PARAM.INC'
      CHARACTER*80 CMNAME,ORNAME
      CHARACTER*3  FLGRAY(15)
      DIMENSION FIELD(NFIELD),STATEV(NSTATV),DIRECT(3,3),
     + T(3,3),TIME(2)
      DIMENSION ARRAY(15),JARRAY(15),JMAC(*),JMATYP(*),COORD(*)
	  
	  
	  !Variable DECLARATION
	  INTEGER	num_elements, i, loc, fieldvalue
	  REAL*8 ELEMENTNR(310376)
	  REAL*8 IPNR(310376)
	  
      common /ELEMENTNRmatrix/ ELEMENTNR                  ! PEMatrix is a common variable
	  common /IPNRmatrix/ IPNR                  ! PEMatrix is a common variable
	  
	  
	  
	  !************************************************************
      fieldvalue=0.D0
      DO i=1,310376
!	   WRITE(7,*) 'i',i 
!       WRITE(7,*) 'ELEMENTNR(i)',ELEMENTNR(i)
       IF (ELEMENTNR(i) .EQ. NOEL) THEN
	     DO j=i,i+7
           IF (IPNR(j) .EQ. NPT) THEN
           !WRITE(7,*) 'NOEL',NOEL
	       fieldvalue=j
		 !         WRITE(7,*) 'fieldvalue',fieldvalue
		   exit
		   ENDIF
		 ENDDO
	   

       exit
	   ENDIF
	  ENDDO
	  
	  FIELD(1)=fieldvalue
	  

	  
	  

	  
      RETURN
      END
	  !**********************************
	  !END of subroutine USDFLD
	  !**********************************
      subroutine uexpan(EXPAN,dexpandt,temp,time,dtime,predef,dpred,
     +     statev,cmname,nstatv,noel)
      include 'aba_param.inc'
      character*80 cmname
      dimension EXPAN(6),dexpandt(*),temp(2),time(2),predef(1),
     +     dpred(*),statev(nstatv)
      !*********************************************************
	  !Variable DECLARATION
	  INTEGER	num_elements, i, loc, fieldvalue
	  REAL*8 PE1(310376)                       ! Size of PEMatrix matrix
	  REAL*8 PE2(310376)                       ! Size of PEMatrix matrix
	  REAL*8 PE3(310376)                       ! Size of PEMatrix matrix
	  REAL*8 PE12(310376)                       ! Size of PEMatrix matrix
	  REAL*8 PE13(310376)                       ! Size of PEMatrix matrix
	  REAL*8 PE23(310376)                       ! Size of PEMatrix matrix
      common /PE1matrix/ PE1               ! PEMatrix is a common variable
      common /PE2matrix/ PE2               ! PEMatrix is a common variable
      common /PE3matrix/ PE3               ! PEMatrix is a common variable
      common /PE12matrix/ PE12               ! PEMatrix is a common variable
      common /PE13matrix/ PE13               ! PEMatrix is a common variable
      common /PE23matrix/ PE23               ! PEMatrix is a common variable
	  IF (TIME(2) .LE. 1.D0) THEN
	    i=PREDEF(1)
	    IF (i .EQ. 0.D0) THEN
	      EXPAN(1)=0.D0
	      EXPAN(2)=0.D0
	      EXPAN(3)=0.D0
	      EXPAN(4)=0.D0
	      EXPAN(5)=0.D0
	      EXPAN(6)=0.D0
        ELSE
   	      EXPAN(1)=PE1(i)
	      EXPAN(2)=PE2(i)
	      EXPAN(3)=PE3(i)
	      EXPAN(4)=PE12(i)
	      EXPAN(5)=PE13(i)
	      EXPAN(6)=PE23(i)
		  !WRITE(7,*) 'EXPAN(1)',EXPAN(1)
        ENDIF
	  
	
      ENDIF


      return
      end subroutine uexpan
	  !**********************************
	  !END of subroutine UEXPAN
	  !**********************************
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,RPL,DDSDDT,
     +     DRPLDE,DRPLDT,STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,
     +     DPRED,CMNAME,NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,
     +     PNEWDT,CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
      INCLUDE 'ABA_PARAM.INC'
      DIMENSION STRESS(NTENS),STATEV(NSTATV),DDSDDE(NTENS,NTENS),
     +     DDSDDT(NTENS),DRPLDE(NTENS),STRAN(NTENS),DSTRAN(NTENS),
     +     TIME(2),PREDEF(1),DPRED(1),PROPS(NPROPS),COORDS(3),DROT(3,3),
     +     DFGRD0(3,3),DFGRD1(3,3)
      CHARACTER*8 CMNAME
      !********************************************************************
      ! Variables defined and used in the UMAT
      INTEGER i,j,k,l
      REAL*8 d_epsm(3,3),alpha1_i(3,3),alpha2_i(3,3),epse_e(3,3),
     +      alpha1_e(3,3),alpha2_e(3,3),epsp_i(3,3),epsp_e(3,3),
     +      sigma_i(3,3),sigma_e(3,3),DTiME,epsp_dot_eq_i,epsp_dot_eq_e,
     +      epsp_eq_i,epsp_eq_e,R_i,R_e,d_epsmtan(3,3),pnewdt_e,depsp_eq_e,
     +      sigmatan_e(3,3),d_sigma_ddepsm(3,3,3,3),Ctens(3,3,3,3),depsp_eq_i,
     +      Mtens(3,3,3,3),ext,mstd,
     +      depsp_eq_n,B_n(3,3),B_an(3,3),
     +      eps_q_e,sml_depsm,TDROT(3,3),JcbnMtrx(3,3,3,3),depsp_eq_e_tan,epse_i(3,3)     
      REAL*8 zero,one,two,MAccur
      PARAMETER(zero=0.D0,one=1.D0,two=2.D0,MAccur=1.D-15)
      !********************************************************************
      ! Time increment should be a positive non-zero value
      DTiME = DTIME
      IF(DTiME .LT. 1.D-15) THEN
        WRITE(*,*) 'DTIME .LT. 1.D-15',DTIME
        WRITE(7,*) 'DTIME .LT. 1.D-15',DTIME
        DTiME = 1.D-10
      ENDIF
      !********************************************************************
	  ! Initialization Substrate
	  IF (TIME(2) .EQ. zero) THEN
	    IF (NSTATV .LT. 30.) WRITE(*,*) 'NSTATV .LT. 30'
        !******************************************************************
        ! Element type check
        IF(NDI.NE.3 .OR. NSHR.NE.3) THEN
          WRITE(*,*) 'Error in Element type for element ',NOEL
          WRITE(7,*) 'Error in Element type for element ',NOEL
        ENDIF
        DO i=1,NSTATV
          STATEV(i) = zero
        ENDDO
	    ext       =      MAX((233.3213211072862D0-1.376984552101672D-1*(TEMP+DTEMP + 273.15D0)),3.D0*one)
        STATEV(28) = PROPS(4)*MAX(PROPS(5)*(one-PROPS(6)*EXP((TEMP+DTEMP + 273.15D0)/PROPS(7))),ext)
		!Must be elastic
		STATEV(28) = STATEV(28)*1000.D0
        mstd     = two
		STATEV(29) = mstd
	  ENDIF
	  
      !********************************************************************
      ! Initialization activated elements
      IF (STATEV(29) .NE. two) THEN
        IF (NSTATV .LT. 30.) WRITE(*,*) 'NSTATV .LT. 30'
        !******************************************************************
        ! Element type check
        IF(NDI.NE.3 .OR. NSHR.NE.3) THEN
          WRITE(*,*) 'Error in Element type for element ',NOEL
          WRITE(7,*) 'Error in Element type for element ',NOEL
        ENDIF
        DO i=1,29
          STATEV(i) = zero
        ENDDO
	    mstd=two
		STATEV(29) = mstd
	    ext       =      MAX((233.3213211072862D0-1.376984552101672D-1*(TEMP+DTEMP + 273.15D0)),3.D0*one)
        STATEV(28) = PROPS(4)*MAX(PROPS(5)*(one-PROPS(6)*EXP((TEMP+DTEMP + 273.15D0)/PROPS(7))),ext)
		!Must be elastic
		STATEV(28) = STATEV(28)*1000.D0
	  ELSE
	  mstd=STATEV(29)
      ENDIF
      !********************************************************************
      ! Import the values of  state variables
      ! First back stress tensor
      alpha1_i(1,1) =  STATEV(1)
      alpha1_i(2,2) =  STATEV(2)
      alpha1_i(3,3) =  STATEV(3)
      alpha1_i(1,2) =  STATEV(4)
      alpha1_i(2,1) =  STATEV(4)
      alpha1_i(1,3) =  STATEV(5)
      alpha1_i(3,1) =  STATEV(5)
      alpha1_i(2,3) =  STATEV(6)
      alpha1_i(3,2) =  STATEV(6)
      ! Second back stress tensor
      alpha2_i(1,1) =  STATEV(7)
      alpha2_i(2,2) =  STATEV(8)
      alpha2_i(3,3) =  STATEV(9)
      alpha2_i(1,2) =  STATEV(10)
      alpha2_i(2,1) =  STATEV(10)
      alpha2_i(1,3) =  STATEV(11)
      alpha2_i(3,1) =  STATEV(11)
      alpha2_i(2,3) =  STATEV(12)
      alpha2_i(3,2) =  STATEV(12)
	  ! Inelastic strain tensor
      epsp_i(1,1) =  STATEV(13)
      epsp_i(2,2) =  STATEV(14)
      epsp_i(3,3) =  STATEV(15)
      epsp_i(1,2) =  STATEV(16)
      epsp_i(2,1) =  STATEV(16)
      epsp_i(1,3) =  STATEV(17)
      epsp_i(3,1) =  STATEV(17)
      epsp_i(2,3) =  STATEV(18)
      epsp_i(3,2) =  STATEV(18)
	  ! Elastic strain
      epse_i(1,1) =  STATEV(19)
      epse_i(2,2) =  STATEV(20)
      epse_i(3,3) =  STATEV(21)
      epse_i(1,2) =  STATEV(22)
      epse_i(2,1) =  STATEV(22)
      epse_i(1,3) =  STATEV(23)
      epse_i(3,1) =  STATEV(23)
      epse_i(2,3) =  STATEV(24)
      epse_i(3,2) =  STATEV(24)
      ! Equivalent inelastic strain
      epsp_eq_i   =  STATEV(25)   
      ! Equivalent inelastic strain rate
      epsp_dot_eq_i   =  STATEV(26)      
	  ! Equivalent inelastic strain increment
	  depsp_eq_i  =  STATEV(27)
      ! Isotropic hardening
      R_i      =  STATEV(28)
	  !********************************************************************
      ! Rotation of the back stresses and  strain tensors
	  ! This is only needed for large deformations
      TDROT    =  TRANSPOSE(DROT)
      alpha1_i =  MATMUL(MATMUL(DROT,alpha1_i),TDROT)
      alpha2_i =  MATMUL(MATMUL(DROT,alpha2_i),TDROT)
      epsp_i   =  MATMUL(MATMUL(DROT,epsp_i)  ,TDROT)
      epse_i   =  MATMUL(MATMUL(DROT,epse_i)  ,TDROT)
      !WRITE(7,*) 'Rotation of back stress and strain tensors'
	  
      !********************************************************************
      ! Get the stress sigma_i(i,i) at the beginning of increment
      ! Get the mechanical strain variation d_epsm(i,i) for the increment
      DO i=1,NDI
        d_epsm(i,i)   = DSTRAN(i)
        sigma_i(i,i)  = STRESS(i)
      ENDDO
      IF (NSHR.NE.0) THEN
        d_epsm(1,2)    = DSTRAN(NDI+1)/two
        d_epsm(2,1)    = d_epsm(1,2)
        sigma_i(1,2)   = STRESS(NDI+1)
        sigma_i(2,1)   = sigma_i(1,2)
        IF (NSHR.NE.1) THEN
          d_epsm(1,3)  = DSTRAN(NDI+2)/two
          d_epsm(3,1)  = d_epsm(1,3)
          sigma_i(1,3) = STRESS(NDI+2)
          sigma_i(3,1) = sigma_i(1,3)
          IF (NSHR.NE.2) THEN
            d_epsm(2,3)  = DSTRAN(NDI+3)/two
            d_epsm(3,2)  = d_epsm(2,3)
            sigma_i(2,3) = STRESS(NDI+3)
            sigma_i(3,2) = sigma_i(2,3)
          ENDIF
        ENDIF
      ENDIF  
	  
        
      !********************************************************************
      ! Material Model
       CALL  MaterialModel(
     +        ! Inputs
     +        DTiME,TEMP,DTEMP,PROPS,NPROPS,TIME,KINC,mstd,
     +        d_epsm,alpha1_i,alpha2_i,epsp_i,depsp_eq_i,
     +        sigma_i,epsp_dot_eq_i,epsp_eq_i,R_i,MAccur,epse_i,
     +        CMNAME,
     +        ! Outputs
     +        pnewdt_e,alpha1_e,alpha2_e,epsp_e,sigma_e,
     +        epsp_dot_eq_e,epsp_eq_e,R_e,epse_e,depsp_eq_e,Ctens)

      !********************************************************************
      ! Suggestion for the next time increment size
      PNEWDT = pnewdt_e
	  
      !********************************************************************
      ! If solution did not converge, reduce the increment size and re-start
      IF (pnewdt_e .LT. one) THEN
        WRITE(7,*) 'NR problem did not converge (1)'
        RETURN    
      ENDIF

      !********************************************************************
      ! Update the stress vector at the end of increment
      DO i=1,NDI
        STRESS(i) = sigma_e(i,i)
      ENDDO
      IF (NSHR.NE.0) THEN
        STRESS(NDI+1)     = sigma_e(1,2)
        IF (NSHR.NE.1) THEN
          STRESS(NDI+2)   = sigma_e(1,3)
          IF (NSHR.NE.2) THEN
            STRESS(NDI+3) = sigma_e(2,3) 
          ENDIF
        ENDIF
      ENDIF
      !********************************************************************
	  ! Update the state variables 
	  ! First back STRESS tensor
      STATEV(1)  = alpha1_e(1,1) 
      STATEV(2)  = alpha1_e(2,2)  
      STATEV(3)  = alpha1_e(3,3)   
      STATEV(4)  = alpha1_e(1,2)    
      STATEV(5)  = alpha1_e(1,3)    
      STATEV(6)  = alpha1_e(2,3)

      ! Second back STRESS tensor
      STATEV(7)  = alpha2_e(1,1)   
      STATEV(8)  = alpha2_e(2,2)   
      STATEV(9)  = alpha2_e(3,3)   
      STATEV(10) = alpha2_e(1,2)  
      STATEV(11) = alpha2_e(1,3)    
      STATEV(12) = alpha2_e(2,3) 

	  ! Inelastic strain
      STATEV(13) = epsp_e(1,1)
      STATEV(14) = epsp_e(2,2)
      STATEV(15) = epsp_e(3,3)     
      STATEV(16) = epsp_e(1,2)    
      STATEV(17) = epsp_e(1,3)   
      STATEV(18) = epsp_e(2,3) 

      ! Elastic strain
      STATEV(19) = epse_e(1,1)
      STATEV(20) = epse_e(2,2)
      STATEV(21) = epse_e(3,3)     
      STATEV(22) = epse_e(1,2)    
      STATEV(23) = epse_e(1,3)   
      STATEV(24) = epse_e(2,3)

      ! Equivalent inelastic strain
      STATEV(25) = epsp_eq_e 

      ! Time-scaled Newton-Raphson solution
      STATEV(26) = epsp_dot_eq_e

	  ! Equivalent inelastic strain increment
	  STATEV(27) = depsp_eq_e
	  
      ! Isotropic hardening
      STATEV(28) = R_e
	  
	  ! Material state
      STATEV(29) = mstd
	  
	  !*****************************************************************
	  ! Determination of the consistent material tangent
      Call JacobianMatrix(
     +        ! Inputs
     +        DTiME,TEMP,DTEMP,PROPS,NPROPS,TIME,NTENS,mstd,
     +        d_epsm,depsp_eq_e,
     +        alpha1_i,alpha2_i,
     +        epsp_dot_eq_e,R_e,R_i,epse_i,
     +        CMNAME,
     +        ! Outputs
     +        Mtens)
	 !********************************************************************
	

      !********************************************************************
      ! Update the tangents in Kelvin-Voigt Notation  
      DDSDDE = zero
	  
      DO i=1,NDI
        DO j=1,NDI
          DDSDDE(i,j) = Mtens(i,i,j,j)
        ENDDO
      ENDDO
      IF (NSHR.NE.0) THEN
        DO i=1,NDI
          DDSDDE(i,NDI+1) = Mtens(i,i,1,2)
          DDSDDE(NDI+1,i) = Mtens(1,2,i,i)
        ENDDO
        DDSDDE(NDI+1,NDI+1) = Mtens(1,2,1,2)
        IF (NSHR.NE.1) THEN
          DO i=1,NDI
            DDSDDE(i,NDI+2) = Mtens(i,i,1,3)
            DDSDDE(NDI+2,i) = Mtens(1,3,i,i)
          ENDDO
          DDSDDE(NDI+2,NDI+2) = Mtens(1,3,1,3)
          DDSDDE(NDI+1,NDI+2) = Mtens(1,2,1,3)
          DDSDDE(NDI+2,NDI+1) = Mtens(1,3,1,2)
          IF (NSHR.NE.2) THEN
            DO i=1,NDI
              DDSDDE(i,NDI+3) = Mtens(i,i,2,3)
              DDSDDE(NDI+3,i) = Mtens(2,3,i,i)
            ENDDO
            DDSDDE(NDI+3,NDI+3) = Mtens(2,3,2,3)
            DDSDDE(NDI+1,NDI+3) = Mtens(1,2,2,3)
            DDSDDE(NDI+3,NDI+1) = Mtens(2,3,1,2)
            DDSDDE(NDI+2,NDI+3) = Mtens(1,3,2,3)
            DDSDDE(NDI+3,NDI+2) = Mtens(2,3,1,3)
          ENDIF
        ENDIF
      ENDIF

	  RETURN
	  
      ENDSUBROUTINE UMAT
	  
	  !**************************************************************************
      SUBROUTINE MaterialModel(
     +        ! Inputs
     +        DTiME,TEMP,DTEMP,PROPS,NPROPS,TIME,KINC,mstd,
     +        d_epsm,alpha1_i,alpha2_i,epsp_i,depsp_eq_i,
     +        sigma_i,epsp_dot_eq_i,epsp_eq_i,R_i,MAccur,epse_i,
     +        CMNAME,
     +        ! Outputs
     +        pnewdt_e,alpha1_e,alpha2_e,epsp_e,sigma_e,
     +        epsp_dot_eq_e,epsp_eq_e,R_e,epse_e,depsp_eq_e,Ctens)
      REAL*8,INTENT(IN)::
     +        DTiME,TEMP,DTEMP,PROPS,TIME,mstd,
     +        d_epsm,alpha1_i,alpha2_i,epsp_i,depsp_eq_i,
     +        sigma_i,epsp_dot_eq_i,epsp_eq_i,R_i,MAccur,epse_i
      INTEGER,INTENT(IN):: NPROPS,KINC
      CHARACTER*8,INTENT(IN):: CMNAME
      REAL*8,INTENT(OUT):: 
     +        pnewdt_e,alpha1_e,alpha2_e,epsp_e,sigma_e,
     +        epsp_dot_eq_e,epsp_eq_e,R_e,epse_e,depsp_eq_e,Ctens
      DIMENSION
     +        ! Inputs
     +        PROPS(NPROPS),TIME(2),d_epsm(3,3),alpha1_i(3,3),
     +        alpha2_i(3,3),epsp_i(3,3),sigma_i(3,3),
     +        epse_i(3,3),
     +        ! Outputs
     +        alpha1_e(3,3),alpha2_e(3,3),epsp_e(3,3),sigma_e(3,3),
     +        epse_e(3,3),Ctens(3,3,3,3)
      !********************************************************************
	  ! Variables defined and used in ‘MaterialModel’
      INTEGER i,j,k,l,counter,flag
      REAL*8 zero,one,two,three
      PARAMETER(zero=0.D0,one=1.D0,two=2.D0,three=3.D0)
      REAL*8 E_modul,nu,A_1,n_1,C_X1,gamma_X1,k_X1,C_X2,gamma_X2,
     +    k_X2,C_R,gamma_R,k_R,Gshear,Kbulk,D_0,
     +    C_X1_old,C_X2_old,C_R_old,PDC_X1,PDC_X2,PDC_R,DQQ,
     +    sigma_tri(3,3),sigma_tri_hyd,output,df_ddepspeq,depsp_eq_near,
     +    sigma_tri_dev(3,3),epsp_dot_eq,Funcion,eror,a,b,t,
     +    depsp(3,3),depsp_eq,TmpK_old,Ddepsp_eq,E_ext,
     +    depsp_eq_u,depsp_eq_l,E_org,gamma_X1ext,gamma_X2ext,
     +    C_R_ext,C_R_ext_old,F_a,F_b,F_0,C_X1_ext,C_X2_ext,C_X1_ext_old,C_X2_ext_old,
     +    Depsp_dot_eq,q,q_tri,beta_tri_dev(3,3),beta_dev(3,3),F_l,F_u,
     +    Iden(3,3),TmpK,depse(3,3),
     +    SAccur,n_tri(3,3),n(3,3)
	 
      !********************************************************************	  
	  ! Material parameters
	  IF (mstd .EQ. one .OR. mstd .EQ. two) THEN
	    IF (TEMP+DTEMP .GT. 1399.5D0) THEN
	    TmpK        =      1399.5D0 + 273.15D0
	    ELSE
	    TmpK        =      TEMP+DTEMP + 273.15D0
	    ENDIF
	  ELSE
	    TmpK        =      1357.6D0 + 273.15D0
	  ENDIF
	  
	  
	  
	  nu  	       =      MIN(MAX(0.3D0, 4.358923362503308D-03*(TmpK-1630.75D0) + 0.3D0),0.482638888888889D0)
      E_ext       =      MIN(MAX((4.382178034116332D5-2.625894854586131D2*TmpK),10000.D0),104000.D0);
      E_modul     =      MAX(PROPS(1)*(one-PROPS(2)*EXP(TmpK/PROPS(3))),E_ext)	 
	  C_R_ext      =      MAX((233.3213211072862D0-1.376984552101672D-1*TmpK),3.D0*one)
	  C_R          =      MAX(PROPS(5)*(one-PROPS(6)*EXP(TmpK/PROPS(7))),C_R_ext)
	  gamma_R      =      PROPS(8) + PROPS(9)/(one+EXP(-PROPS(10)*(TmpK-PROPS(11))))
	  k_R          =      PROPS(12)+PROPS(13)/(one+EXP(-PROPS(14)*(TmpK-PROPS(15))))
	  C_X1_ext     =      170.D0*2541.5D0+1255.D0 + 710000.D0*0.376D0/(one+EXP(0.0189D0*(TmpK-1375.D0)))
	  C_X1         =      MIN(PROPS(16)*(one-PROPS(17)*EXP(TmpK/PROPS(18))),C_X1_ext)
	  C_X2_ext     =      70.D0*1147.65D0+540.D0 + 140000.D0*0.43D0/(one+EXP(0.01545D0*(TmpK-1370.D0)))
	  C_X2         =      MIN(PROPS(19)*(one-PROPS(20)*EXP(TmpK/PROPS(21))),C_X2_ext)
	  gamma_X1ext  =      MAX(zero,3.2D-1*0.21558D0*(TmpK-1273.15D0)**3.D0)
      gamma_X1     =      PROPS(22) + PROPS(23)*EXP(TmpK/PROPS(24)) + gamma_X1ext
	  gamma_X2ext  =      MAX(zero,8.D-2*0.215160D0*(TmpK-1273.15D0)**3.D0)
      gamma_X2     =      PROPS(25) + PROPS(26)*EXP(TmpK/PROPS(27)) + gamma_X2ext
	  k_X1         =      PROPS(28)/(one + EXP(-PROPS(29)*(TmpK-PROPS(30))))
	  k_X2         =      PROPS(31)/(one + EXP(-PROPS(32)*(TmpK-PROPS(33))))
	  A_1          =      PROPS(34)*EXP(TmpK/PROPS(35))
	  n_1          =      PROPS(36) + PROPS(37)/(one + EXP(PROPS(38)*(TmpK-PROPS(39))))
	  R_0          =      PROPS(4)*C_R
	  

	  
	  !Variables for non-isothermal terms
	  IF (mstd .EQ. two) THEN
	    IF (TEMP .GT. 1399.5D0) THEN
	    TmpK_old        =      1399.5D0 + 273.15D0
	    ELSE
	    TmpK_old        =      TEMP + 273.15D0
	    ENDIF
	  ELSEIF (mstd .EQ. one .OR. mstd .EQ. -one) THEN
	    TmpK_old        =      1357.6D0 + 273.15D0
	  ENDIF
	  
	  C_X1_ext_old      =      170.D0*2541.5D0+1255.D0 + 710000.D0*0.376D0/(one+EXP(0.0189D0*(TmpK_old-1375.D0)))
	  C_X1_old          =      MIN(PROPS(16)*(one-PROPS(17)*EXP(TmpK_old/PROPS(18))),C_X1_ext_old)
	  C_X2_ext_old      =      70.D0*1147.65D0+540.D0 + 140000.D0*0.43D0/(one+EXP(0.01545D0*(TmpK_old-1370.D0)))
	  C_X2_old          =      MIN(PROPS(19)*(one-PROPS(20)*EXP(TmpK_old/PROPS(21))),C_X2_ext_old)
	  C_R_ext_old       =      MAX((233.3213211072862D0-1.376984552101672D-1*TmpK_old),3.D0*one)
	  C_R_old           =      MAX(PROPS(5)*(one-PROPS(6)*EXP(TmpK_old/PROPS(7))),C_R_ext_old)
	  PDC_X1             =     (C_X1-C_X1_old)/C_X1
	  PDC_X2             =     (C_X2-C_X2_old)/C_X2
	  PDC_R              =     (C_R-C_R_old)/C_R
	  
	  
	  !********************************************************************
	  ! Strain accuracy
      SAccur      = 1.D-10
	  
      !********************************************************************
	  ! Identity matrix (3*3)
      DO i=1,3
        DO j=1,3
	    IF (i .EQ. j) THEN
            Iden(i,j) = 1.D0
          ELSE
            Iden(i,j) = 0.D0
          END IF
        END DO
      ENDDO

      !********************************************************************
      ! Shear and Bulk moduli
      Gshear = E_modul/(two*(one+nu))
      Kbulk  = E_modul/(three*(one-two*nu))

	  !********************************************************************
      ! Elastic stiffness tensor Ctens
      Ctens = zero
      DO i=1,3
        DO j=1,3
          DO k=1,3
             DO l=1,3
               Ctens(i,j,k,l) = Ctens(i,j,k,l) 
     +         +Gshear*(Iden(i,k)*Iden(j,l)+Iden(i,l)*Iden(j,k)
     +         -(two/three)*Iden(i,j)*Iden(k,l))
     +         + Kbulk*Iden(i,j)*Iden(k,l)
             ENDDO
          ENDDO
        ENDDO
      ENDDO

      !********************************************************************
      ! Trial sigma
      sigma_tri = zero
      DO i=1,3
        DO j=1,3
          DO k=1,3
             DO l=1,3
               sigma_tri(i,j) = sigma_tri(i,j)
     +         +Ctens(i,j,k,l)*(d_epsm(k,l)+epse_i(k,l))
             ENDDO
          ENDDO
        ENDDO
      ENDDO

      !********************************************************************
      ! Deviatoric and hydrostatic parts of trial sigma
      sigma_tri_hyd =(sigma_tri(1,1)+sigma_tri(2,2)+sigma_tri(3,3))/3.D0
      sigma_tri_dev = sigma_tri - Iden * sigma_tri_hyd

      !********************************************************************
      ! Newton-Raphson (NR) solution for determination of variables
      counter     = 0
	  flag        = 0
      eror        = one
	  F_u         = HUGE(one)
	  F_l         = -HUGE(one)

      !********************************************************************
      !Elastic solution
      depsp_eq     =  zero
      Ddepsp_eq_old     =  zero

      !********************************************************************
      DO WHILE (eror.GT.SAccur .OR. ISNAN(eror) .OR. depsp_eq.LT.zero)
        !******************************************************************
        ! Use of NR solutions of last increment as initial guess  
            IF (counter .EQ. 1) THEN
			   SAccur = 1.D-10
			   IF (depsp_eq_i .EQ. zero) THEN
			      depsp_eq = one*5.D-6
			       ! WRITE(7,*) 'previous increment is elastic',depsp_eq
			   ELSE
	              depsp_eq=epsp_dot_eq_i*DTiME
			       ! WRITE(7,*) 'guess from previous increment rate',epsp_dot_eq_i*DTiME
			   ENDIF
            ENDIF

        !****************************************************************
		! counter > 1 Elastic prediction was not accepted. Apply NR Method
	IF (counter    .GT. 1   ) THEN	    
			
		  !****************************************************************
          ! Calculation of Depsp_dot_eq
           Ddepsp_eq   = -Funcion/df_ddepspeq
		   
          !****************************************************************
		  ! Update of equivalent viscoplastic strain increment
          ! NR iteration or switch to bisection method for counter>49
		    IF (counter .EQ. 50) THEN
				 IF(F_u.EQ. HUGE(one)) THEN
				   depsp_eq_u=9.D-2*one
				 ENDIF
				 IF(F_l.EQ. -HUGE(one)) THEN
				   depsp_eq_l=9.D-2*one
				 ENDIF
				 a=zero
				 F_a=F_0
				 IF (F_0 .GT. zero) THEN
				  F_b=F_l
				  b=MIN(depsp_eq_l,9.D-2*one)
				 ELSE
				  F_b=F_u
				  b=MIN(depsp_eq_u,9.D-2*one)
				 ENDIF
				 t=(b-a)/two + a
				 depsp_eq = t
            ELSEIF (counter .GT. 50) THEN
		   	      IF(Funcion .LT. zero) THEN
			        IF(F_a .LT. zero) THEN
			           a=t
				    ELSE
				       b=t
				    ENDIF
			      ELSEIF (Funcion .GT. zero) THEN
			        IF(F_a .GT. zero) THEN
			    	    a=t
				    ELSE
			           b=t
				    ENDIF
			      ENDIF
			      t=(b-a)/two + a
			      depsp_eq = t
			ELSE
		          depsp_eq     = depsp_eq + Ddepsp_eq
            ENDIF
			
		    Ddepsp_eq_old=Ddepsp_eq
	 ENDIF

		!******************************************************************
        ! Calculation of function values
		Funcion = zero 
		
        !******************************************************************
        ! Calculation of Function value
        CALL FUNCTIO(
     +         ! Inputs
     +         depsp_eq,DTiME,TEMP,DTEMP,Gshear,Kbulk,Ctens,sigma_tri_dev,
     +         alpha1_i,alpha2_i,C_X1,gamma_X1,k_X1,A_1,C_X2,gamma_X2,
     +         PDC_X1,PDC_X2,PDC_R,DQQ,counter,mstd,
     +         k_X2,D_0,R_i,C_R,gamma_R,k_R,n_1,iden,
     +         ! Outputs
     +         output,q_tri,q,R_e,beta_tri_dev,df_ddepspeq)
	 
	    ! Residual
        Funcion  =  output

        !******************************************************************
	    ! Calculation of Error
        eror=ABS(Funcion)

        !******************************************************************
		!Update F_u and F_l only during NR iterations
		IF (counter .EQ. 0 .AND. depsp_eq .GE. zero) THEN
		   F_0=Funcion
		ELSEIF (counter .LT. 50 .AND. depsp_eq .GE. zero) THEN
		IF (Funcion .GT. zero .AND. Funcion .LT. F_u) THEN
		   F_u=Funcion
		   depsp_eq_u=depsp_eq
		ENDIF
	    IF (Funcion .LT. zero .AND. Funcion .GT. F_l) THEN
		   F_l=Funcion
		   depsp_eq_l=depsp_eq
		ENDIF
		
		ENDIF

        ! Convergence check
        IF (counter .GT. 100) THEN
          pnewdt_e = one/4.D0
           WRITE(7,*) 'Counter gt 200. Convergence is judged unlikely,CUTBACK',Funcion,depsp_eq,Funcion_near,depsp_eq_near
          RETURN
        ENDIF
		
		! Update counter
        counter=counter+1

      ENDDO
	  
      !********************************************************************
      ! Update variables with accepted equivalent viscoplastic strain increment
	  
      !********************************************************************
      ! Calculation of equivalent inelastic strain rate
	  epsp_dot_eq = depsp_eq/DTiME
	  
      !********************************************************************
      ! Calculation of beta_dev
      beta_dev = beta_tri_dev/(one+(three*Gshear*epsp_dot_eq*DTiME)/q
     +       +(C_X1*epsp_dot_eq*DTiME)/(q*(one+gamma_X1*epsp_dot_eq*DTiME+k_X1*DTiME-PDC_X1))
     +       +(C_X2*epsp_dot_eq*DTiME)/(q*(one+gamma_X2*epsp_dot_eq*DTiME+k_X2*DTiME-PDC_X2)))
	 
      !********************************************************************
	  ! Calculation of the viscoplastic flow direction n
          IF (q .EQ. zero) THEN
            n  =  zero
          ELSE
	        n=(three/two)*(beta_dev/q)
          ENDIF

	  !********************************************************************
	  ! Calculation of the trial viscoplastic flow direction ntr
          IF (q_tri .EQ. zero) THEN
            n_tri  =  zero
          ELSE
	        n_tri=(three/two)*(beta_tri_dev/q_tri)
          ENDIF

	  !********************************************************************
	  ! Calculation of inelastic strain increments
	  depsp=n*epsp_dot_eq*DTiME

	  !********************************************************************
	  ! Calculation of elastic strain increments
	  depse=d_epsm-depsp


      !********************************************************************
      ! Update back stress
      alpha1_e = (alpha1_i/(one+gamma_X1*epsp_dot_eq*DTiME+k_X1*DTiME-PDC_X1))+
     +          ((two/three)*C_X1*epsp_dot_eq*DTiME*n/(one+gamma_X1*epsp_dot_eq*DTiME+k_X1*DTiME-PDC_X1))
      alpha2_e = (alpha2_i/(one+gamma_X2*epsp_dot_eq*DTiME+k_X2*DTiME-PDC_X2))+
     +          ((two/three)*C_X2*epsp_dot_eq*DTiME*n/(one+gamma_X2*epsp_dot_eq*DTiME+k_X2*DTiME-PDC_X2))

      !********************************************************************
      ! Update other strain variables
      epsp_e      =  depsp        + epsp_i
      epse_e      =  depse        + epse_i
      epsp_eq_e   =  depsp_eq     + epsp_eq_i
      epsp_dot_eq_e   =  epsp_dot_eq
	  depsp_eq_e = depsp_eq

      !********************************************************************
      ! Update stress tensor
      sigma_e = zero
      DO i=1,3
        DO j=1,3
          DO k=1,3
             DO l=1,3
               sigma_e(i,j) = sigma_e(i,j) + Ctens(i,j,k,l)*(epse_e(k,l))
             ENDDO
          ENDDO
        ENDDO
      ENDDO

	  !********************************************************************
      ! Determination of a suggestion for next time-increment-size  
	  IF(depsp_eq .GT. 9.D-1) THEN
	         pnewdt_e = (9.D-1 / (depsp_eq))
			 ! WRITE(7,*) 'depsp_eq',depsp_eq
	          ! WRITE(7,*) 'depsp_eq to big: pnewdt_e',pnewdt_e, depsp_eq, TEMP
	  ELSEIF (depsp_eq .GT. 1.D-7 .AND. depsp_eq .LT. 9.D-1) THEN
		     pnewdt_e = two - (depsp_eq-1.D-7)/(9.D-1 - 1.D-7)
		     !pnewdt_e = one
	          ! WRITE(7,*) 'depsp_eq in between: pnewdt_e',pnewdt_e
			 ! WRITE(7,*) 'depsp_eq',depsp_eq
	  ELSE
	    
             pnewdt_e = two

	  ENDIF
	  
      RETURN
      !********************************************************************
      ENDSUBROUTINE MaterialModel
	  
	  
      !**************************************************************************
      SUBROUTINE FUNCTIO(
     +         ! Inputs
     +         depsp_eq,DTiME,TEMP,DTEMP,Gshear,Kbulk,Ctens,sigma_tri_dev,
     +         alpha1_i,alpha2_i,C_X1,gamma_X1,k_X1,A_1,C_X2,gamma_X2,
     +         PDC_X1,PDC_X2,PDC_R,DQQ,counter,mstd,
     +         k_X2,D_0,R_i,C_R,gamma_R,k_R,n_1,iden,
     +         ! Outputs
     +         output,q_tri,q,R_e,beta_tri_dev,df_ddepspeq)
      REAL*8,INTENT(IN):: 
     +         depsp_eq,DTiME,TEMP,DTEMP,Gshear,Kbulk,Ctens,sigma_tri_dev,
     +         alpha1_i,alpha2_i,C_X1,gamma_X1,k_X1,A_1,C_X2,gamma_X2,
     +         PDC_X1,PDC_X2,PDC_R,DQQ,mstd,
     +         k_X2,D_0,R_i,C_R,gamma_R,k_R,n_1,iden
      INTEGER,INTENT(IN):: counter
      REAL*8,INTENT(OUT):: 
     +         output,q_tri,q,R_e,beta_tri_dev,df_ddepspeq
      DIMENSION 
     +         ! Inputs
     +         alpha1_i(3,3),alpha2_i(3,3),
     +         Ctens(3,3,3,3),sigma_tri_dev(3,3),iden(3,3), 
     +         ! Outputs
     +         beta_tri_dev(3,3)
      !********************************************************************
      ! Variables defined and used in the FUNCTION1
      INTEGER i,j,k,l
      REAL*8 zero,one,two,three
      PARAMETER(zero=0.D0,one=1.D0,two=2.D0,three=3.D0)
      REAL*8 alpha1_i_hyd,alpha2_i_hyd,sigma_eff,
     +  dq_ddepspeq,dR_ddepspeq,epsp_dot_eq, 
     +  alpha1_i_dev(3,3),alpha2_i_dev(3,3), 
     +  dbetatridev_ddepspeq(3,3),dqtri_dbetatridev(3,3)
	 
      !********************************************************************
      ! Deviatoric and hydrostatic part of input back stress tensors 
      alpha1_i_hyd =(alpha1_i(1,1)+alpha1_i(2,2)+alpha1_i(3,3))/3.D0
      alpha2_i_hyd =(alpha2_i(1,1)+alpha2_i(2,2)+alpha2_i(3,3))/3.D0
      alpha1_i_dev = alpha1_i - Iden * alpha1_i_hyd
      alpha2_i_dev = alpha2_i - Iden * alpha2_i_hyd

      !********************************************************************
      ! Trial beta deviatoric
      beta_tri_dev = sigma_tri_dev -(
     + (alpha1_i_dev/(one+gamma_X1*depsp_eq+k_X1*DTiME-PDC_X1))+
     + (alpha2_i_dev/(one+gamma_X2*depsp_eq+k_X2*DTiME-PDC_X2)))

      !********************************************************************
      ! Trial q
      q_tri = SQRT((three/two)*SUM(beta_tri_dev*beta_tri_dev))

      !********************************************************************
      ! Determination of q
      IF (q_tri .EQ. zero) THEN
        q   = zero
      ELSE
        q   =   q_tri -(three*Gshear*depsp_eq) -(
     +       (C_X1*depsp_eq)/(one+gamma_X1*depsp_eq+k_X1*DTiME-PDC_X1)+
     +       (C_X2*depsp_eq)/(one+gamma_X2*depsp_eq+k_X2*DTiME-PDC_X2))
      ENDIF

	  !********************************************************************
      ! Radius of yield surface
	  IF (mstd .EQ. -one) THEN
	    R_e  =  R_i
	  ELSE 
        R_e  =  MAX((R_i + C_R*depsp_eq)/(one+gamma_R*depsp_eq+k_R*DTiME-PDC_R),zero)
      ENDIF
      !********************************************************************
	  ! Viscous stress
       sigma_eff = MAX((q - R_e),zero)

      !********************************************************************
      ! Equivalent viscoplastic strain rate
	  epsp_dot_eq = A_1*sigma_eff**n_1
	  
      !********************************************************************
      ! NR Residual
      output = depsp_eq - epsp_dot_eq*DTiME
	  
	  !**************************************************************
	  ! Analytical calculation of residual gradient
	  
	  !********************************************************************  
	  ! Partial derivative of the deviatoric relative trial stress 
	  ! with respect to the equivalent viscoplastic strain increment
	  dbetatridev_ddepspeq=zero
	  dbetatridev_ddepspeq=(
     +   +(alpha1_i_dev*gamma_X1/(one+gamma_X1*depsp_eq+k_X1*DTiME-PDC_X1)**2)
     +   +(alpha2_i_dev*gamma_X2/(one+gamma_X2*depsp_eq+k_X2*DTiME-PDC_X2)**2))

	  
	  !********************************************************************  
	  ! Partial derivative of trial q
	  ! with respect to the deviatoric relative trial stress
	  dqtri_dbetatridev = zero
	  IF (q_tri .EQ. zero) then
	  dqtri_dbetatridev = zero
	  ELSE
	  dqtri_dbetatridev = one/(two*q_tri)*three*Beta_tri_dev
	  ENDIF
	  
	  !********************************************************************  
	  ! Partial derivative of trial q
	  ! with respect to the equivalent viscoplastic strain increment
	  dqtri_ddepspeq=SUM(dqtri_dbetatridev*dbetatridev_ddepspeq)
	  
	  !********************************************************************  
	  ! Partial derivative of q
	  ! with respect to the equivalent viscoplastic strain increment
      dq_ddepspeq=dqtri_ddepspeq - three*Gshear -(
     +       (C_X1*(one+gamma_X1*depsp_eq+k_X1*DTiME-PDC_X1)-(gamma_X1*C_X1*depsp_eq))/
     +       (one+gamma_X1*depsp_eq+k_X1*DTiME-PDC_X1)**2+
     +       (C_X2*(one+gamma_X2*depsp_eq+k_X2*DTiME-PDC_X2)-(gamma_X2*C_X2*depsp_eq))/
     +       (one+gamma_X2*depsp_eq+k_X2*DTiME-PDC_X2)**2)
	   
	  !********************************************************************  
	  ! Partial derivative of the isotropic hardening
	  ! with respect to the equivalent viscoplastic strain increment
	  IF (mstd .EQ. -one) THEN
	  dR_ddepspeq=zero
	  ELSE
	  dR_ddepspeq=C_R/(one+gamma_R*depsp_eq+k_R*DTiME-PDC_R)-gamma_R*(R_i + C_R*depsp_eq)
     +                      /(one+gamma_R*depsp_eq+k_R*DTiME-PDC_R)**2
	  ENDIF
	  
	  !********************************************************************  
	  ! Residual gradient
	  df_ddepspeq= one-A_1*n_1*(MAX(q-R_e,zero))**(n_1-one)*DTiME*(dq_ddepspeq-dR_ddepspeq)

	  IF (ISNAN(df_ddepspeq)) THEN
        WRITE(7,*) 'df_ddepspeq is NAN: ',df_ddepspeq

	  ENDIF
	  

      RETURN
      ENDSUBROUTINE FUNCTIO
!**************************************************************************




!**************************************************************************
      SUBROUTINE JacobianMatrix(
     +        ! Inputs
     +        DTiME,TEMP,DTEMP,PROPS,NPROPS,TIME,NTENS,mstd,
     +        d_epsm,depsp_eq_e,
     +        alpha1_i,alpha2_i,
     +        epsp_dot_eq_e,R_e,R_p,epse_i,
     +        CMNAME,
     +        ! Outputs
     +        JcbnMtrx)
      REAL*8,INTENT(IN)::
     +        DTiME,TEMP,DTEMP,PROPS,TIME,mstd,
     +        d_epsm,depsp_eq_e,
     +        alpha1_i,alpha2_i,
     +        epsp_dot_eq_e,R_e,R_p,epse_i
      INTEGER,INTENT(IN):: NTENS,NPROPS
      CHARACTER*8,INTENT(IN):: CMNAME
      REAL*8,INTENT(OUT):: 
     +        JcbnMtrx
      DIMENSION
     +        ! Inputs
     +        PROPS(NPROPS),TIME(2),d_epsm(3,3),
     +        alpha1_i(3,3),alpha2_i(3,3),
     +        epse_i(3,3),
     +        ! Outputs
     +        JcbnMtrx(3,3,3,3)
      !********************************************************************
	  ! Variables defined and used in ‘JacobianMatrix’
      INTEGER i,j,k,l,m,o,counter,flag
      REAL*8 zero,one,two,three
      PARAMETER(zero=0.D0,one=1.D0,two=2.D0,three=3.D0)
      REAL*8 E_modul,nu,A_1,n_1,C_X1,gamma_X1,k_X1,C_X2,gamma_X2,
     +    k_X2,C_R,gamma_R,k_R,Gshear,Kbulk,
     +    C_X1_old,C_X2_old,C_R_old,PDC_X1,PDC_X2,PDC_R,
     +    alpha1_i_hyd,alpha2_i_hyd,dqtri_ddepspeq,dq_ddepspeq,
     +    T,df_ddepspeq,Btens(3,3),
     +    C_R_ext,C_R_ext_old,C_X1_ext,C_X2_ext,C_X1_ext_old,C_X2_ext_old,
     +    dR_ddepspeq,alpha1_i_dev(3,3),alpha2_i_dev(3,3),
     +    sigma_tri(3,3),sigma_tri_hyd,betadev_m_betadev,
     +    sigma_tri_dev(3,3),epsp_dot_eq,dsigtri_depsm(3,3,3,3),
     +    depsp_eq,TmpK_old,E_ext,
     +    q,q_tri,beta_tri_dev(3,3),beta_dev(3,3),
     +    Iden(3,3),TmpK,dsigtridev_dsigtri(3,3,3,3),
     +    Stens(3,3,3,3),Ptens(3,3,3,3),Qtens(3,3,3,3),
     +    dbetadev_dq(3,3),dn_dsigtri(3,3,3,3),dqtri_dsigtri(3,3),Ltens(3,3),
     +    dbetatridev_ddepspeq(3,3),dntri_ddepspeq(3,3),df_dsigtri(3,3),
     +    n_tri(3,3),n(3,3),dqtri_dbetatridev(3,3),Ctens(3,3,3,3)
      !********************************************************************
	  
      !********************************************************************
	  !Updated equivalent viscoplastic strain increment and rate 
	  depsp_eq=depsp_eq_e
	  epsp_dot_eq=epsp_dot_eq_e
	  
      !********************************************************************
	  ! Material parameters
	  IF (mstd .EQ. one .OR. mstd .EQ. two) THEN
	    IF (TEMP+DTEMP .GT. 1399.5D0) THEN
	    TmpK        =      1399.5D0 + 273.15D0
	    ELSE
	    TmpK        =      TEMP+DTEMP + 273.15D0
	    ENDIF
	  ELSE
	    TmpK        =      1357.6D0 + 273.15D0
	  ENDIF
	  
	  
	  
	  nu  	       =      MIN(MAX(0.3D0, 4.358923362503308D-03*(TmpK-1630.75D0) + 0.3D0),0.482638888888889D0)
      E_ext       =      MIN(MAX((4.382178034116332D5-2.625894854586131D2*TmpK),10000.D0),104000.D0);
      E_modul     =      MAX(PROPS(1)*(one-PROPS(2)*EXP(TmpK/PROPS(3))),E_ext)	 
	  C_R_ext      =      MAX((233.3213211072862D0-1.376984552101672D-1*TmpK),3.D0*one)
	  C_R          =      MAX(PROPS(5)*(one-PROPS(6)*EXP(TmpK/PROPS(7))),C_R_ext)
	  gamma_R      =      PROPS(8) + PROPS(9)/(one+EXP(-PROPS(10)*(TmpK-PROPS(11))))
	  k_R          =      PROPS(12)+PROPS(13)/(one+EXP(-PROPS(14)*(TmpK-PROPS(15))))
	  C_X1_ext     =      170.D0*2541.5D0+1255.D0 + 710000.D0*0.376D0/(one+EXP(0.0189D0*(TmpK-1375.D0)))
	  C_X1         =      MIN(PROPS(16)*(one-PROPS(17)*EXP(TmpK/PROPS(18))),C_X1_ext)
	  C_X2_ext     =      70.D0*1147.65D0+540.D0 + 140000.D0*0.43D0/(one+EXP(0.01545D0*(TmpK-1370.D0)))
	  C_X2         =      MIN(PROPS(19)*(one-PROPS(20)*EXP(TmpK/PROPS(21))),C_X2_ext)
	  gamma_X1ext  =      MAX(zero,3.2D-1*0.21558D0*(TmpK-1273.15D0)**3.D0)
      gamma_X1     =      PROPS(22) + PROPS(23)*EXP(TmpK/PROPS(24)) + gamma_X1ext
	  gamma_X2ext  =      MAX(zero,8.D-2*0.215160D0*(TmpK-1273.15D0)**3.D0)
      gamma_X2     =      PROPS(25) + PROPS(26)*EXP(TmpK/PROPS(27)) + gamma_X2ext
	  k_X1         =      PROPS(28)/(one + EXP(-PROPS(29)*(TmpK-PROPS(30))))
	  k_X2         =      PROPS(31)/(one + EXP(-PROPS(32)*(TmpK-PROPS(33))))
	  A_1          =      PROPS(34)*EXP(TmpK/PROPS(35))
	  n_1          =      PROPS(36) + PROPS(37)/(one + EXP(PROPS(38)*(TmpK-PROPS(39))))
	  R_0          =      PROPS(4)*C_R
	  

	  
	  !Variables for non-isothermal terms
	  IF (mstd .EQ. two) THEN
	    IF (TEMP .GT. 1399.5D0) THEN
	    TmpK_old        =      1399.5D0 + 273.15D0
	    ELSE
	    TmpK_old        =      TEMP + 273.15D0
	    ENDIF
	  ELSEIF (mstd .EQ. one .OR. mstd .EQ. -one) THEN
	    TmpK_old        =      1357.6D0 + 273.15D0
	  ENDIF
	  
	  C_X1_ext_old      =      170.D0*2541.5D0+1255.D0 + 710000.D0*0.376D0/(one+EXP(0.0189D0*(TmpK_old-1375.D0)))
	  C_X1_old          =      MIN(PROPS(16)*(one-PROPS(17)*EXP(TmpK_old/PROPS(18))),C_X1_ext_old)
	  C_X2_ext_old      =      70.D0*1147.65D0+540.D0 + 140000.D0*0.43D0/(one+EXP(0.01545D0*(TmpK_old-1370.D0)))
	  C_X2_old          =      MIN(PROPS(19)*(one-PROPS(20)*EXP(TmpK_old/PROPS(21))),C_X2_ext_old)
	  C_R_ext_old       =      MAX((233.3213211072862D0-1.376984552101672D-1*TmpK_old),3.D0*one)
	  C_R_old           =      MAX(PROPS(5)*(one-PROPS(6)*EXP(TmpK_old/PROPS(7))),C_R_ext_old)
	  PDC_X1             =     (C_X1-C_X1_old)/C_X1
	  PDC_X2             =     (C_X2-C_X2_old)/C_X2
	  PDC_R              =     (C_R-C_R_old)/C_R
	  

	  !********************************************************************
	  ! Identity matrix (3*3)
      DO i=1,3
        DO j=1,3
	    IF (i .EQ. j) THEN
            Iden(i,j) = 1.D0
          ELSE
            Iden(i,j) = 0.D0
          END IF
        END DO
      ENDDO
	  
	  !********************************************************************
      ! Shear and Bulk moduli
      Gshear = E_modul/(two*(one+nu))
      Kbulk  = E_modul/(three*(one-two*nu))
	  
	  !********************************************************************
      ! Elastic stiffness tensor Ctens
      Ctens = zero
      DO i=1,3
        DO j=1,3
          DO k=1,3
             DO l=1,3
               Ctens(i,j,k,l) = Ctens(i,j,k,l) 
     +         +Gshear*(Iden(i,k)*Iden(j,l)+Iden(i,l)*Iden(j,k)
     +         -(two/three)*Iden(i,j)*Iden(k,l))
     +         + Kbulk*Iden(i,j)*Iden(k,l)
             ENDDO
          ENDDO
        ENDDO
      ENDDO
	  
	  !********************************************************************
      ! Trial sigma
      sigma_tri = zero
      DO i=1,3
        DO j=1,3
          DO k=1,3
             DO l=1,3
               sigma_tri(i,j) = sigma_tri(i,j)
     +         +Ctens(i,j,k,l)*(d_epsm(k,l)+epse_i(k,l))
             ENDDO
          ENDDO
        ENDDO
      ENDDO
	  
	  !********************************************************************
      ! Trial deviatoric sigma
      sigma_tri_hyd =(sigma_tri(1,1)+sigma_tri(2,2)+sigma_tri(3,3))/3.D0
      sigma_tri_dev = sigma_tri - Iden * sigma_tri_hyd
	  
	  !********************************************************************
      ! Deviatoric and hydrostatic part of previous back stress tensors 
      alpha1_i_hyd =(alpha1_i(1,1)+alpha1_i(2,2)+alpha1_i(3,3))/3.D0
      alpha2_i_hyd =(alpha2_i(1,1)+alpha2_i(2,2)+alpha2_i(3,3))/3.D0
      alpha1_i_dev = alpha1_i - Iden * alpha1_i_hyd
      alpha2_i_dev = alpha2_i - Iden * alpha2_i_hyd
	  
	  !********************************************************************
	  ! Deviatoric relative trial stress
      beta_tri_dev = sigma_tri_dev -(
     + (alpha1_i_dev/(one+gamma_X1*depsp_eq+k_X1*DTiME-PDC_X1))+
     + (alpha2_i_dev/(one+gamma_X2*depsp_eq+k_X2*DTiME-PDC_X2)))
	 
	 !********************************************************************
      ! Von Mises equivalent of the trial relative stress qtri
      q_tri = SQRT((three/two)*SUM(beta_tri_dev*beta_tri_dev))
	  
	  !********************************************************************
	  ! Von Mises equivalent of the relative stress q
      IF (q_tri .EQ. zero) THEN
        q   = zero
      ELSE
        q   =   q_tri -(three*Gshear*depsp_eq) -(
     +       (C_X1*depsp_eq)/(one+gamma_X1*depsp_eq+k_X1*DTiME-PDC_X1)+
     +       (C_X2*depsp_eq)/(one+gamma_X2*depsp_eq+k_X2*DTiME-PDC_X2))
      ENDIF
	  
	  
	  !********************************************************************  
	  ! Consistent material tangent matrix
	  JcbnMtrx = zero
	  IF (q - R_e .LE. zero) then
	  JcbnMtrx=Ctens
	  ELSE
         !********************************************************************
         ! Multiplicative viscoplastic corrector T
         IF (q .EQ. zero) THEN
         T=zero
         ELSE
         T=one/(one+(three*Gshear*epsp_dot_eq*DTiME)/q
     +       +(C_X1*epsp_dot_eq*DTiME)/(q*(one+gamma_X1*epsp_dot_eq*DTiME+k_X1*DTiME-PDC_X1))
     +       +(C_X2*epsp_dot_eq*DTiME)/(q*(one+gamma_X2*epsp_dot_eq*DTiME+k_X2*DTiME-PDC_X2)))
         ENDIF
         
         !********************************************************************
         ! Deviatoric relativ stress
         beta_dev = beta_tri_dev*T
        
         !********************************************************************
         ! Viscoplastic flow direction n
             IF (q .EQ. zero) THEN
               n  =  zero
             ELSE
               n=(three/two)*(beta_dev/q)
             ENDIF
        
         !********************************************************************  
         ! Partial derivative of the deviatoric trial stress
         ! with respect to the trial stress
         dsigtridev_dsigtri = zero
          DO i=1,3
            DO j=1,3
              DO k=1,3
                 DO l=1,3
                    dsigtridev_dsigtri(i,j,k,l) = dsigtridev_dsigtri(i,j,k,l) 
     +                +(one)*(Iden(i,k)*Iden(j,l))        
     +                + (-one/three)*Iden(i,j)*Iden(k,l)
                 ENDDO
              ENDDO
            ENDDO
          ENDDO
          
         !********************************************************************  
         ! Partial derivative of the trial stress
         ! with respect to the incremental total strain tensor
         dsigtri_depsm=Ctens
        
         !********************************************************************  
         ! Partial derivative of trial q
         ! with respect to the deviatoric relative trial stress
         dqtri_dbetatridev = zero
         IF (q_tri .EQ. zero) then
         dqtri_dbetatridev = zero
         ELSE
         dqtri_dbetatridev = one/(two*q_tri)*three*Beta_tri_dev
         ENDIF
         
         !********************************************************************  
         ! Double contraction of the deviatoric relative stress tensor with itself	
         betadev_m_betadev=SUM(beta_tri_dev*beta_tri_dev)
         
         !********************************************************************  
         ! Partial derivative of the deviatoric relative stress
         ! with respect to q
         dbetadev_dq=zero
         
         !********************************************************************  
         ! Partial derivative of trial q
         ! with respect to the trial stress tensor
         dqtri_dsigtri=zero
          DO i=1,3
            DO j=1,3
              DO k=1,3
                 DO l=1,3
                 dqtri_dsigtri(i,j)=dqtri_dsigtri(i,j)  
     +             +dqtri_dbetatridev(k,l)*dsigtridev_dsigtri(k,l,i,j)
                 ENDDO
              ENDDO
            ENDDO
          ENDDO
        
         !********************************************************************  
         ! Partial derivative of the viscoplastic flow direction
         ! with respect to the trial stress tensor
         dn_dsigtri=zero
         IF (q_tri .EQ. zero) THEN
         dn_dsigtri=zero
         ELSE
          DO i=1,3
            DO j=1,3
              DO k=1,3
                 DO l=1,3
                 dn_dsigtri(i,j,k,l)=dn_dsigtri(i,j,k,l)
     +             +((three/two)*q_tri*dsigtridev_dsigtri(i,j,k,l)
     +             -(three/two)*dqtri_dsigtri(i,j)*beta_tri_dev(k,l))/(q_tri**2)
                 ENDDO
              ENDDO
            ENDDO
          ENDDO
          ENDIF
         
         !********************************************************************  
         ! Partial derivative of the deviatoric relative trial stress
         ! with respect to the equivalent viscoplastic strain increment
         dbetatridev_ddepspeq=zero
         dbetatridev_ddepspeq=(
     +     +(alpha1_i_dev*gamma_X1/(one+gamma_X1*depsp_eq+k_X1*DTiME-PDC_X1)**2)
     +     +(alpha2_i_dev*gamma_X2/(one+gamma_X2*depsp_eq+k_X2*DTiME-PDC_X2)**2))
         
         !********************************************************************  
         ! Partial derivative of the deviatoric relative trial stress
         ! with respect to the equivalent viscoplastic strain increment
         dqtri_ddepspeq=SUM(dqtri_dbetatridev*dbetatridev_ddepspeq)
         
         !********************************************************************  
         ! Partial derivative of the trial viscoplastic flow direction
         ! with respect to the equivalent viscoplastic strain increment
         dntri_ddepspeq=zero
         IF (q_tri .EQ. zero) THEN
         dntri_ddepspeq=zero
         ELSE
          DO i=1,3
            DO j=1,3
             dntri_ddepspeq(i,j)=dntri_ddepspeq(i,j)+
     +         ((three/two)*q_tri*dbetatridev_ddepspeq(i,j)
     +         -(three/two)*beta_tri_dev(i,j)*dqtri_ddepspeq)/q_tri**2
            ENDDO
          ENDDO
          ENDIF
          
         !********************************************************************  
         ! Partial derivative of q
         ! with respect to the equivalent viscoplastic strain increment
         dq_ddepspeq=dqtri_ddepspeq - three*Gshear -(
     +       (C_X1*(one+gamma_X1*depsp_eq+k_X1*DTiME-PDC_X1)-(gamma_X1*C_X1*depsp_eq))/
     +       (one+gamma_X1*depsp_eq+k_X1*DTiME-PDC_X1)**2+
     +       (C_X2*(one+gamma_X2*depsp_eq+k_X2*DTiME-PDC_X2)-(gamma_X2*C_X2*depsp_eq))/
     +       (one+gamma_X2*depsp_eq+k_X2*DTiME-PDC_X2)**2)
          
         !********************************************************************  
         ! Partial derivative of the isotropic hardening
         ! with respect to the equivalent viscoplastic strain increment
         IF (mstd .EQ. -one) then
         dR_ddepspeq=zero
         ELSE
         dR_ddepspeq=(C_R*(one+gamma_R*depsp_eq+k_R*DTiME-PDC_R)-gamma_R*(R_p + C_R*depsp_eq))
     +                      /(one+gamma_R*depsp_eq+k_R*DTiME-PDC_R)**2
         ENDIF
          
         
         !********************************************************************  
         ! Residual gradient
        
         df_ddepspeq= one-A_1*n_1*(q-R_e)**(n_1-one)*DTiME*(dq_ddepspeq-dR_ddepspeq) 
          
         !********************************************************************  
         ! Partial derivative of the NR residual function
         ! with respect to the trial stress tensor
         
         df_dsigtri=-A_1*n_1*(q-R_e)**(n_1-one)*DTiME*dqtri_dsigtri
        
         !********************************************************************  
         ! Ltens=df_dsigtri : Ctens
         Ltens=zero
          DO i=1,3
            DO j=1,3
              DO k=1,3
                 DO l=1,3
                    Ltens(i,j)=Ltens(i,j)+df_dsigtri(k,l)*Ctens(k,l,i,j)
                 ENDDO
              ENDDO
            ENDDO
          ENDDO
        
         !********************************************************************  
         ! Btens=-Ltens/df_ddepspeq
         
         IF (df_ddepspeq .EQ. zero) then
         Btens=zero
         ELSE
         Btens=-Ltens/df_ddepspeq
         ENDIF
        
         !********************************************************************  
         ! Stens=dn_dsigtri:dsigtri_depsm
         Stens=zero
          DO i=1,3
            DO j=1,3
              DO k=1,3
                 DO l=1,3
                   DO m=1,3
        		  DO o=1,3
        		    Stens(i,j,k,l)=Stens(i,j,k,l)
     +                +dn_dsigtri(i,j,m,o)*dsigtri_depsm(m,o,k,l)
                     ENDDO
                   ENDDO
                 ENDDO
              ENDDO
            ENDDO
          ENDDO
        
         !********************************************************************  
         ! Ptens=n x Btens + depsp_eq*(df_ddepspeq x Btens + Stens)
         Ptens=zero
          DO i=1,3
            DO j=1,3
              DO k=1,3
                 DO l=1,3
                    Ptens(i,j,k,l)=n(i,j)*Btens(k,l)+depsp_eq*(dntri_ddepspeq(i,j)*Btens(k,l)+Stens(i,j,k,l))
                 ENDDO
              ENDDO
            ENDDO
          ENDDO
        
         !********************************************************************  
         ! Viscoplastic stiffness correction tensor Qtens=Ctens:Ptens
         Qtens=zero
          DO i=1,3
            DO j=1,3
              DO k=1,3
                 DO l=1,3
                   DO m=1,3
        		  DO o=1,3
        		    Qtens(i,j,k,l)=Qtens(i,j,k,l)
     +                +Ctens(i,j,m,o)*Ptens(m,o,k,l)
                     ENDDO
                   ENDDO
                 ENDDO
              ENDDO
            ENDDO
          ENDDO

	  JcbnMtrx=Ctens-Qtens
	  ENDIF

	  RETURN
      !********************************************************************
      ENDSUBROUTINE JacobianMatrix
      !**************************************************************************