using LinearAlgebra
using QuantumOptics
using WignerSymbols

function Kfactor(F,I=1.5,J=0.5)
    return F*(F+1)-I*(I+1)-J*(J+1)
end

function LandeFactor(J,L=0,S=0.5)
    return 1+(J*(J+1)+S*(S+1)-L*(L+1))/(2*J*(J+1))
end

struct CGmatrix{T<:Rational} 
    cgMatrix::Array{Float64}
    function CGmatrix{T}(j1::T,j2::T) where T<:Rational
        if !(( j1.den == 1 || j1.den == 2 ) && ( j2.den == 1  || j2.den ==2 ))
            error("spins are not half-integer")
        end

        row=1
        col=1
        dims = Int64((j1*2+1) * (j2*2+1))
        cgMatrix=zeros(Float64,dims,dims)
        j3max=maximum((abs(j1+j2), abs(j1-j2)))
        j3min=minimum((abs(j1+j2), abs(j1-j2)))
        for j3=j3max:-1:j3min
            for m3=j3:-1:-j3
                for m2=j2:-1:-j2
                    for m1=j1:-1:-j1
                        cgMatrix[row,col]=Float64(clebschgordan(j1,m1,j2,m2,j3,m3))
                        col+=1
                    end

                end
                row+=1
                col=1
            end
        end
        new(cgMatrix)
    end
end
CGmatrix(a::T,b::T) where T<: Rational = CGmatrix{T}(a,b)

function getCGmatrix(self::CGmatrix)
    return self.cgMatrix
end

struct HyperfineManifold{T<:Rational} 
    fmBaseKets::Array{Ket}
    fvalues::Array{Float64}
    function HyperfineManifold{T}(spinI::T,spinJ::T) where T<:Rational 
        if !(( spinI.den == 1 || spinI.den == 2 ) && ( spinJ.den == 1  || spinJ.den ==2 ))
            error("spins are not half-integer")
        end



        mIbasis=SpinBasis(spinI)                        
        mJbasis=SpinBasis(spinJ)
        mImJbasis=CompositeBasis(mIbasis,mJbasis)

        mIbasisSz= 0.5*sigmaz(mIbasis)               # Sz operator for S1 basis: 0.5 \hbar [1 0; 0 -1]
        mJbasisSz = 0.5*sigmaz(mJbasis)               # Sz operator for S2 basis: 0.5 \hbar [1 0; 0 -1]
        mIbasisS₊ = sigmap(mIbasis)                # Splus operator for S1 basis: \hbar [0 1; 0  0]
        mIbasisS₋ = sigmam(mIbasis)               # Sminus operator for S1 basis: \hbar [0 0; 1  0]
        mJbasisS₊ = sigmap(mJbasis)                # Splus operator for S2 basis: \hbar [0 1; 0  0]
        mJbasisS₋ = sigmam(mJbasis)               # Sminus operator for S1 basis: \hbar [0 0; 1  0]
        mIbasisIdentity = identityoperator(mIbasis)   # Identity operator for S1 basis: [1 0; 0 1]
        mJbasisIdentity = identityoperator(mJbasis)   # Identity operator for S2 basis: [1 0; 0 1]
        mIbasisS² = spinI*(spinI+1)*mIbasisIdentity              # S^2 operator for S1 basis: I(I+1) \hbar^2 [1 0; 0 1]
        mJbasisS² = spinJ*(spinJ+1)*mJbasisIdentity              # S^2 operator for S2 basis: J(J+1) \hbar^2 [1 0; 0 1]

        Jz=tensor(mIbasisIdentity,mJbasisSz)     # S1z in S1S2 basis 
        Iz=tensor(mIbasisSz,mJbasisIdentity)     # S2z in S1S2 basis
        J₊=tensor(mIbasisIdentity,mJbasisS₊)     # S1plus in S1S2 basis 
        J₋=tensor(mIbasisIdentity,mJbasisS₋)     # S1minus in S1S2 basis
        I₊=tensor(mIbasisS₊,mJbasisIdentity)     # S2plus in S1S2 basis 
        I₋=tensor(mIbasisS₋,mJbasisIdentity)     # S2minus in S1S2 basis
        mImJbasisI²=tensor(mIbasisS²,mJbasisIdentity)
        mImJbasisJ²=tensor(mIbasisIdentity,mJbasisS²)
        IJ=Iz*Jz+0.5*(I₊*J₋+I₋*J₊)        # S1̇ S2 in S1S2 basis
        mImJbasisF²= mImJbasisI² + mImJbasisJ² + 2*Iz*Jz +I₊*J₋ +I₋*J₊
        
        C=getCGmatrix(CGmatrix(spinI,spinJ))
        
        TransformMatrix=DenseOperator(mImJbasis,C)
        TransformMatrixDagger=dagger(TransformMatrix)

        Izfmbasis=TransformMatrix*(Iz)*TransformMatrixDagger    # Eq. E-10 in Chapter XII.E.2.a. of vol. 2 of Cohen-Tannoudji, Quantum Mechanics.
        Jzfmbasis=TransformMatrix*(Jz)*TransformMatrixDagger

        K1=Kfactor(1)
        K2=Kfactor(2)
        FMhfsDiagonal=Diagonal([K2, K2, K2, K2, K2, K1, K1, K1])
        IJfmbasis2=0.5*DenseOperator(mImJbasis,FMhfsDiagonal)

        fmBaseKets=Array{Ket}(undef,length(mImJbasis))
        fvalues=Array{Rational}(undef, length(mImJbasis))
        mvalues=Array{Rational}(undef, length(mImJbasis))

        for i = 1:length(mImJbasis)
            fmBaseKets[i] = Ket(mImJbasis,C[i,:])
            mᵢ = real(dagger(fmBaseKets[i])*Iz*fmBaseKets[i])
            mⱼ = real(dagger(fmBaseKets[i])*Jz*fmBaseKets[i])
            mf = mᵢ+mⱼ
            F² = real(dagger(fmBaseKets[i])*mImJbasisF²*fmBaseKets[i])
            F = (-1 + sqrt(1+4*F²))/2
            fvalues[i] = Int64(round(2*F))//2
            mvalues[i] = Int64(round(2*mf))//2
        end

        new(fmBaseKets,fvalues)
    end
end
HyperfineManifold(a::T,b::T) where T<: Rational = HyperfineManifold{T}(a,b)