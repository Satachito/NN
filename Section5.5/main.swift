import Foundation

import Accelerate

struct
Matrix {
	var	nR	:	Int
	var	nC	:	Int
	var	m	:	[ Double ]

	init( _ nR: Int, _ nC: Int, _ m: ArraySlice< Double > ) {
		guard nR * nC == m.count else { fatalError( "Size unmatch" ) }
		self.nR = nR
		self.nC = nC
		self.m = Array( m )
	}

	init( _ nR: Int, _ nC: Int, _ initial: Double = 0 ) {
		self.nR = nR
		self.nC = nC
		self.m = [ Double ]( repeating: initial, count: nR * nC )
	}

	init( _ p: [ [ Double ] ] ) {
		guard p.count > 0 else { fatalError( "Need to have at least one element." ) }
		self.init( p.count, p[ 0 ].count )
		for i in 0 ..< nR {
			guard p[ i ].count == nC else { fatalError( "All the rows must have same size." ) }
			let	w = i * nC
			m[ w ..< w + nC ] = ArraySlice( p[ i ] )
		}
	}

	subscript( r: Int, c: Int ) -> Double {
		return m[ r * nC + c ];
	}
}

func
Dump( _ p: Matrix ) {
	for iR in 0 ..< p.nR {
		for iC in 0 ..< p.nC {
			print( p[ iR, iC ], terminator: "\t" )
		}
		print()
	}
}

prefix func
~ ( p: Matrix ) -> Matrix {
	var	v = Matrix( p.nC, p.nR )
	vDSP_mtransD( p.m, 1, &v.m, 1, vDSP_Length( v.nR ), vDSP_Length( v.nC ) )
	return v
}

func
Dot( _ l: Matrix, _ r: Matrix ) -> Matrix {
	guard l.nC == r.nR else { fatalError() }
	var	v = Matrix( l.nR, r.nC )
	vDSP_mmulD( l.m, 1, r.m, 1, &v.m, 1, vDSP_Length( l.nR ), vDSP_Length( r.nC ), vDSP_Length( l.nC ) )
	return v
}

func
HAdd( _ l: Matrix, _ r: [ Double ] ) -> Matrix {
	guard l.nC == r.count else { fatalError() }
	var	v = Matrix( l.nR, l.nC )
	for iR in 0 ..< v.nR {
		let	wOffset = l.nC * iR
		vDSP_vaddD( UnsafePointer( l.m ) + wOffset, 1, r, 1, &v.m + wOffset, 1, vDSP_Length( v.nC ) )
	}
	return v
}

func
HSum( _ p: Matrix ) -> [ Double ] {
	var	v = [ Double ]( repeating: 0, count: p.nC )
	for iR in 0 ..< p.nR {
		let	wOffset = p.nC * iR
		vDSP_vaddD( UnsafePointer( p.m ) + wOffset, 1, v, 1, &v, 1, vDSP_Length( v.count ) )
	}
	return v
}

struct
ReLU {
	var	o = [ Double ]()
	mutating func
	Forward( _ p: [ Double ] ) -> [ Double ] {
		o = p.map { $0 < 0 ? 0 : $0 }
		return o
	}
	func
	Backward( _ d: [ Double ] ) -> [ Double ] {
		return ( 0 ..< d.count ).map { o[ $0 ] > 0 ? d[ $0 ] : 0 }
	}
}

struct
Sigmoid {
	var	o = [ Double ]()
	mutating func
	Forward( _ p: [ Double ] ) -> [ Double ] {
		o = p.map { 1 / ( 1 + exp( -$0 ) ) }
		return o
	}
	func
	Backward( _ d: [ Double ] ) -> [ Double ] {
		return ( 0 ..< d.count ).map { p -> Double in d[ p ] * ( o[ p ] * ( 1 - o[ p ] ) ) }
	}
}

struct
Affine {
	var	W	: Matrix
	var	b	: [ Double ]
	var	dW	: Matrix
	var	db	: [ Double ]

	var	o	: Matrix

	mutating func
	Forward( _ p: Matrix ) -> Matrix {
		o = HAdd( Dot( p, W ), b )
		return o
	}
	mutating func
	Backward( _ d: Matrix, _ p: Matrix ) -> Matrix {
		let v = Dot( d, ~W )
		dW = Dot( ~p, d )
		db = HSum( d )
		return v
	}
}
