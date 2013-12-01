//****************************************************************************************************************//

	This repositry shows the way the SALSA20_8 double round can be implemented using AVX enabled processors 
	The compiled output will be close to the Assembly files as have been provided in the original repository 
	from "https://github.com/pooler/cpuminer" The single way mining speed of this code is equal to the 3way 
	variant as the original repository supplies in Assembly.

	THIS REPOSITORY SHOULD NOT BE USED IF YOU AIM IS JUST MINING. 

	The reson for this setup was to see if an effective algoritm for the salsa algoritm could also 
	be written in C instead of using (hard to maintain) assemby code. Some of my hard coded adaptations will 
	decrease effictiveness if your computer is not equal to my development setup (Intel i3, Ubuntu)

	Below you can find an highlevel overview of the 'changes' to the algoritm in order to 
	support a parallel salsa20 doubleround.

	Have fun, Gerko de Roo
		
//****************************************************************************************************************//
	
Parallel usage of the SIDM registers(SSE, AVX, AVX2) for the salsa doubleround:

The array of 16 32 bit inputs can be better represented in an array, in order to analyse the
calculation sequence:

Input Matrix (4x4):

|       x00     x01     x02     x03 |
|       x04     x05     x06     x07 |
|       x08     x09     x0a     x0b |
|       x0c     x0d     x0e     x0f |

Calculations:

The capital X notes a new value as function of two matix values, and a rotary shift

The first calculation with a rotairy shift of 7

|       x00             x01             x02             X03(x0f,x0b,7)  |
|       X04(x00,x0c,7)  x05             x06             x07             |
|       x08             X09(x05,x01,7)  x0a             x0b             |
|       x0c             x0d             X0e(x0a,x06,7)  x0f             |

The second calculation with a rotary shift of 9

|       x00             x01             X02(X0e,x0a,9)  X03             |
|       X04             x05             x06             X07(X03,x0f,9)  |
|       X08(X04,x00,9)  X09             x0a             x0b             |
|       x0c             X0d(X09,x05,9)  X0e             x0f             |

The third calculation with a rotary shift of 13

|       x00             X01(X0d,X09,13) X02             X03             |
|       X04             x05             X06(X02,X0e,13) X07             |
|       X08             X09             x0a             X0b(X07,X03,13) |
|       X0c(X08,X04,13) X0d             X0e             x0f             |

The fourth calculation with a rotary shift of 18

|       X00(X0c,X08,18) X01             X02             X03             |
|       X04             X05(X01,X0d,18) X06             X07             |
|       X08             X09             X0a(X06,X02,18) X0b             |
|       X0c             X0d             X0e             X0f(X0b,X07,18) |


Dependency in this calucaltion is only in the colums, not in the rows.
This enables the possibility to rearange the matrix so that calculations can be done per full row:

|       x04     x09     x0e     x03 |
|       x08     x0d     x02     x07 |
|       x0c     x01     x06     x0b |
|       x00     x05     x0a     x0f |

This will change the calculation to:


|       X04(x00,x0c,7)  X09(x05,x01,7)  X0e(x0a,x06,7)  X03(x0f,x0b,7)  |
|       x08             x0d             x02             x07             |
|       x0c             x01             x06             x0b             |
|       x00             x05             x0a             x0f             |

(etc for the other rows)


Using the SIDM the complete row can be calculated with 4 value's in parallel by the 128 SIDM bit registers.
In the second round the caclulation needs to be done in columns.
By transposing the matrix, the same routine can be used for both the columns and row calculation
for a complete double round.

Transposed matrix for paralel columns:

|       x01     x06     x0b     x0c |
|       x02     x07     x08     x0d |
|       x03     x04     x09     x0e |
|       x00     x05     x0a     x0f |


The transpose is actually a rotairy shift:

	row1 shift >> 1 (32 bits)
	row2 shift >> 2 (32 bits)
	row3 shift << 1 (32 bits)
	and the rows need to be reordered.
		Row 1 becomes row 3
		row 3 becomes row 1

When a cpu containes 256 bit SIDM registors, two salsa20-8 calculations could be made full parallel.



