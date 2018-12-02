//
//  main.swift
//  Predict
//
//  Created by Satoru Ogura on 2018/11/22.
//  Copyright © 2018 Satoru Ogura. All rights reserved.
//

import Foundation

let	w = try! DecodeJSON( Data( contentsOf: URL( string: "file:///Users/sat/Desktop/NN/sample_weight.json" )! ) ) as! JSONDict

func
MakeMatrix< F: Numeric >( _ p: [ [ F ] ] ) -> Matrix< F > {
	var	v = Matrix< F >( p.count, p[ 0 ].count );
	for i in 0 ..< v.nR {
		guard p[ i ].count == v.nC else { fatalError() }
		v.SetRow( i, ArraySlice( p[ i ] ) )
	}
	return v
}

let	W1 = MakeMatrix( w[ "W1" ] as! [ [ Double ] ] )
let	W2 = MakeMatrix( w[ "W2" ] as! [ [ Double ] ] )
let	W3 = MakeMatrix( w[ "W3" ] as! [ [ Double ] ] )

let	b1 = ArraySlice( w[ "b1" ] as![ Double ] )
let	b2 = ArraySlice( w[ "b2" ] as![ Double ] )
let	b3 = ArraySlice( w[ "b3" ] as![ Double ] )

print( W1.nR, W1.nC )
print( W2.nR, W2.nC )
print( W3.nR, W3.nC )

print( b1.count )
print( b2.count )
print( b3.count )

W1.u.withUnsafeBufferPointer { p in
	let	wData = Data( buffer: p )
	wData.withUnsafeBytes { ( p: UnsafePointer< Double > ) in
		print( p[ 0 ] )
	}
}

//let	wOF = FileHandle( forWritingAtPath: "/Users/sat/Desktop/NN/sample_weight.bin" )!
//W1.u.withUnsafeBufferPointer { p in wOF.write( Data( buffer: p ) ) }
//W2.u.withUnsafeBufferPointer { p in wOF.write( Data( buffer: p ) ) }
//W3.u.withUnsafeBufferPointer { p in wOF.write( Data( buffer: p ) ) }
//Array( b1 ).withUnsafeBufferPointer { p in wOF.write( Data( buffer: p ) ) }
//Array( b2 ).withUnsafeBufferPointer { p in wOF.write( Data( buffer: p ) ) }
//Array( b3 ).withUnsafeBufferPointer { p in wOF.write( Data( buffer: p ) ) }
//wOF.closeFile()

let	wIF = FileHandle( forReadingAtPath: "/Users/sat/Desktop/NN/sample_weight.bin" )!
let	wData = wIF.readData( ofLength: 8 )
wData.withUnsafeBytes { ( p: UnsafePointer< Double > ) in
	print( p[ 0 ] )
}

/*
var	wDouble = [ 0.0 ];
wDouble.withUnsafeMutableBufferPointer { ( p: inout UnsafeMutableBufferPointer<Double> ) in
}
*/
