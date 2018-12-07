import Foundation

import Accelerate

struct
ReLU {
	var	o = ArraySlice< Double >()
	mutating func
	Forward( _ p: ArraySlice< Double > ) -> ArraySlice< Double > {
		o = ArraySlice( p.map { $0 < 0 ? 0 : $0 } )
		return o
	}
	func
	Backward( _ d: ArraySlice< Double > ) -> ArraySlice< Double > {
		return ArraySlice( ( 0 ..< d.count ).map { o[ $0 ] > 0 ? d[ $0 ] : 0 } )
	}
}

struct
Sigmoid {
	var	o = ArraySlice< Double >()
	mutating func
	Forward( _ p: ArraySlice< Double > ) -> ArraySlice< Double > {
		o = ArraySlice( p.map { 1 / ( 1 + exp( -$0 ) ) } )
		return o
	}
	func
	Backward( _ d: ArraySlice< Double > ) -> ArraySlice< Double > {
//		return ( 0 ..< d.count ).map { p -> Double in d[ p ] * ( o[ p ] * ( 1 - o[ p ] ) ) }
		return ArraySlice( ( 0 ..< d.count ).map { p -> Double in d[ p ] * ( o[ p ] * ( 1 - o[ p ] ) ) } )
	}
}

struct
Affine< N: Numeric > {
	var	W	: Matrix< N >
	var	b	: ArraySlice< N >
	var	dW	: Matrix< N >
	var	db	: ArraySlice< N >

	var	o	: Matrix< N >

	mutating func
	Forward( _ p: Matrix< N > ) -> Matrix< N > {
		o = HAdd( Dot( p, W ), b )
		return o
	}
	mutating func
	Backward( _ d: Matrix< N >, _ p: Matrix< N > ) -> Matrix< N > {
		let v = Dot( d, ~W )
		dW = Dot( ~p, d )
		db = HSum( d )
		return v
	}
}

func
Softmax( _ p: ArraySlice< Double > ) {
	let	v = Exp( p - Max( ArraySlice( p ) ) )
	return v / Sum( v )
}

print( Softmax( [ 0.3, 2.9, 4.0 ] ))
struct
SoftmaxWithLoss {
	var	o = ArraySlice< Double >()
	mutating func
	Forward( _ p: ArraySlice< Double >, _ t: ArraySlice< Double > ) -> ArraySlice< Double > {
		o = Softmax( p )
		return o
	}
	mutating func
	Backward( _ d: ArraySlice< Double >, _ t: ArraySlice< Double > ) -> ArraySlice< Double > {
		return ( o - t ) / d.count
	}
}
