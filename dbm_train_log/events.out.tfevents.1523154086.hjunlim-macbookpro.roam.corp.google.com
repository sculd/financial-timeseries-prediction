       �K"	  �)_��Abrain.Event:2�k��4     �[	�8�)_��A"��
g
dataPlaceholder*
dtype0*
shape:���������*'
_output_shapes
:���������
i
labelsPlaceholder*
dtype0*
shape:���������*'
_output_shapes
:���������
g
truncated_normal/shapeConst*
dtype0*
valueB"   �   *
_output_shapes
:
Z
truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
\
truncated_normal/stddevConst*
dtype0*
valueB
 *�А=*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:	�
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*
_output_shapes
:	�
n
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes
:	�
~
Variable
VariableV2*
dtype0*
shape:	�*
	container *
shared_name *
_output_shapes
:	�
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
:	�
j
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*
_output_shapes
:	�
`
zeros/shape_as_tensorConst*
dtype0*
valueB:�*
_output_shapes
:
P
zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
i
zerosFillzeros/shape_as_tensorzeros/Const*

index_type0*
T0*
_output_shapes	
:�
x

Variable_1
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Variable_1/AssignAssign
Variable_1zeros*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
i
truncated_normal_1/shapeConst*
dtype0*
valueB"�   d   *
_output_shapes
:
\
truncated_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_1/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:	�d
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*
_output_shapes
:	�d
t
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*
_output_shapes
:	�d
�

Variable_2
VariableV2*
dtype0*
shape:	�d*
	container *
shared_name *
_output_shapes
:	�d
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
:	�d
p
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*
_output_shapes
:	�d
a
zeros_1/shape_as_tensorConst*
dtype0*
valueB:d*
_output_shapes
:
R
zeros_1/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
n
zeros_1Fillzeros_1/shape_as_tensorzeros_1/Const*

index_type0*
T0*
_output_shapes
:d
v

Variable_3
VariableV2*
dtype0*
shape:d*
	container *
shared_name *
_output_shapes
:d
�
Variable_3/AssignAssign
Variable_3zeros_1*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:d
k
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes
:d
i
truncated_normal_2/shapeConst*
dtype0*
valueB"d   2   *
_output_shapes
:
\
truncated_normal_2/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_2/stddevConst*
dtype0*
valueB
 *��>*
_output_shapes
: 
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes

:d2
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*
_output_shapes

:d2
s
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*
_output_shapes

:d2
~

Variable_4
VariableV2*
dtype0*
shape
:d2*
	container *
shared_name *
_output_shapes

:d2
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:d2
o
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes

:d2
a
zeros_2/shape_as_tensorConst*
dtype0*
valueB:2*
_output_shapes
:
R
zeros_2/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
n
zeros_2Fillzeros_2/shape_as_tensorzeros_2/Const*

index_type0*
T0*
_output_shapes
:2
v

Variable_5
VariableV2*
dtype0*
shape:2*
	container *
shared_name *
_output_shapes
:2
�
Variable_5/AssignAssign
Variable_5zeros_2*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes
:2
k
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes
:2
i
truncated_normal_3/shapeConst*
dtype0*
valueB"2      *
_output_shapes
:
\
truncated_normal_3/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_3/stddevConst*
dtype0*
valueB
 *.�d>*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes

:2
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes

:2
s
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes

:2
~

Variable_6
VariableV2*
dtype0*
shape
:2*
	container *
shared_name *
_output_shapes

:2
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes

:2
o
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes

:2
a
zeros_3/shape_as_tensorConst*
dtype0*
valueB:*
_output_shapes
:
R
zeros_3/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
n
zeros_3Fillzeros_3/shape_as_tensorzeros_3/Const*

index_type0*
T0*
_output_shapes
:
v

Variable_7
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
�
Variable_7/AssignAssign
Variable_7zeros_3*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:
i
truncated_normal_4/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
\
truncated_normal_4/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_4/stddevConst*
dtype0*
valueB
 *�5?*
_output_shapes
: 
�
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes

:
�
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0*
_output_shapes

:
s
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0*
_output_shapes

:
~

Variable_8
VariableV2*
dtype0*
shape
:*
	container *
shared_name *
_output_shapes

:
�
Variable_8/AssignAssign
Variable_8truncated_normal_4*
validate_shape(*
_class
loc:@Variable_8*
use_locking(*
T0*
_output_shapes

:
o
Variable_8/readIdentity
Variable_8*
_class
loc:@Variable_8*
T0*
_output_shapes

:
a
zeros_4/shape_as_tensorConst*
dtype0*
valueB:*
_output_shapes
:
R
zeros_4/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
n
zeros_4Fillzeros_4/shape_as_tensorzeros_4/Const*

index_type0*
T0*
_output_shapes
:
v

Variable_9
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
�
Variable_9/AssignAssign
Variable_9zeros_4*
validate_shape(*
_class
loc:@Variable_9*
use_locking(*
T0*
_output_shapes
:
k
Variable_9/readIdentity
Variable_9*
_class
loc:@Variable_9*
T0*
_output_shapes
:
~
MatMulMatMuldataVariable/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:����������
V
addAddMatMulVariable_1/read*
T0*(
_output_shapes
:����������
T
LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
]
LeakyRelu/mulMulLeakyRelu/alphaadd*
T0*(
_output_shapes
:����������
c
LeakyRelu/MaximumMaximumLeakyRelu/muladd*
T0*(
_output_shapes
:����������
V
dropout/keep_probConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
^
dropout/ShapeShapeLeakyRelu/Maximum*
out_type0*
T0*
_output_shapes
:
_
dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
_
dropout/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:����������
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:����������
p
dropout/addAdddropout/keep_probdropout/random_uniform*
T0*(
_output_shapes
:����������
V
dropout/FloorFloordropout/add*
T0*(
_output_shapes
:����������
o
dropout/divRealDivLeakyRelu/Maximumdropout/keep_prob*
T0*(
_output_shapes
:����������
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:����������
�
MatMul_1MatMuldropout/mulVariable_2/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������d
Y
add_1AddMatMul_1Variable_3/read*
T0*'
_output_shapes
:���������d
V
LeakyRelu_1/alphaConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
b
LeakyRelu_1/mulMulLeakyRelu_1/alphaadd_1*
T0*'
_output_shapes
:���������d
h
LeakyRelu_1/MaximumMaximumLeakyRelu_1/muladd_1*
T0*'
_output_shapes
:���������d
X
dropout_1/keep_probConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
b
dropout_1/ShapeShapeLeakyRelu_1/Maximum*
out_type0*
T0*
_output_shapes
:
a
dropout_1/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
a
dropout_1/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:���������d
�
dropout_1/random_uniform/subSubdropout_1/random_uniform/maxdropout_1/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_1/random_uniform/mulMul&dropout_1/random_uniform/RandomUniformdropout_1/random_uniform/sub*
T0*'
_output_shapes
:���������d
�
dropout_1/random_uniformAdddropout_1/random_uniform/muldropout_1/random_uniform/min*
T0*'
_output_shapes
:���������d
u
dropout_1/addAdddropout_1/keep_probdropout_1/random_uniform*
T0*'
_output_shapes
:���������d
Y
dropout_1/FloorFloordropout_1/add*
T0*'
_output_shapes
:���������d
t
dropout_1/divRealDivLeakyRelu_1/Maximumdropout_1/keep_prob*
T0*'
_output_shapes
:���������d
f
dropout_1/mulMuldropout_1/divdropout_1/Floor*
T0*'
_output_shapes
:���������d
�
MatMul_2MatMuldropout_1/mulVariable_4/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������2
Y
add_2AddMatMul_2Variable_5/read*
T0*'
_output_shapes
:���������2
V
LeakyRelu_2/alphaConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
b
LeakyRelu_2/mulMulLeakyRelu_2/alphaadd_2*
T0*'
_output_shapes
:���������2
h
LeakyRelu_2/MaximumMaximumLeakyRelu_2/muladd_2*
T0*'
_output_shapes
:���������2
X
dropout_2/keep_probConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
b
dropout_2/ShapeShapeLeakyRelu_2/Maximum*
out_type0*
T0*
_output_shapes
:
a
dropout_2/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
a
dropout_2/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:���������2
�
dropout_2/random_uniform/subSubdropout_2/random_uniform/maxdropout_2/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_2/random_uniform/mulMul&dropout_2/random_uniform/RandomUniformdropout_2/random_uniform/sub*
T0*'
_output_shapes
:���������2
�
dropout_2/random_uniformAdddropout_2/random_uniform/muldropout_2/random_uniform/min*
T0*'
_output_shapes
:���������2
u
dropout_2/addAdddropout_2/keep_probdropout_2/random_uniform*
T0*'
_output_shapes
:���������2
Y
dropout_2/FloorFloordropout_2/add*
T0*'
_output_shapes
:���������2
t
dropout_2/divRealDivLeakyRelu_2/Maximumdropout_2/keep_prob*
T0*'
_output_shapes
:���������2
f
dropout_2/mulMuldropout_2/divdropout_2/Floor*
T0*'
_output_shapes
:���������2
�
MatMul_3MatMuldropout_2/mulVariable_6/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
Y
add_3AddMatMul_3Variable_7/read*
T0*'
_output_shapes
:���������
V
LeakyRelu_3/alphaConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
b
LeakyRelu_3/mulMulLeakyRelu_3/alphaadd_3*
T0*'
_output_shapes
:���������
h
LeakyRelu_3/MaximumMaximumLeakyRelu_3/muladd_3*
T0*'
_output_shapes
:���������
X
dropout_3/keep_probConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
b
dropout_3/ShapeShapeLeakyRelu_3/Maximum*
out_type0*
T0*
_output_shapes
:
a
dropout_3/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
a
dropout_3/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:���������
�
dropout_3/random_uniform/subSubdropout_3/random_uniform/maxdropout_3/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_3/random_uniform/mulMul&dropout_3/random_uniform/RandomUniformdropout_3/random_uniform/sub*
T0*'
_output_shapes
:���������
�
dropout_3/random_uniformAdddropout_3/random_uniform/muldropout_3/random_uniform/min*
T0*'
_output_shapes
:���������
u
dropout_3/addAdddropout_3/keep_probdropout_3/random_uniform*
T0*'
_output_shapes
:���������
Y
dropout_3/FloorFloordropout_3/add*
T0*'
_output_shapes
:���������
t
dropout_3/divRealDivLeakyRelu_3/Maximumdropout_3/keep_prob*
T0*'
_output_shapes
:���������
f
dropout_3/mulMuldropout_3/divdropout_3/Floor*
T0*'
_output_shapes
:���������
�
MatMul_4MatMuldropout_3/mulVariable_8/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
Y
add_4AddMatMul_4Variable_9/read*
T0*'
_output_shapes
:���������
V
LeakyRelu_4/alphaConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
b
LeakyRelu_4/mulMulLeakyRelu_4/alphaadd_4*
T0*'
_output_shapes
:���������
h
LeakyRelu_4/MaximumMaximumLeakyRelu_4/muladd_4*
T0*'
_output_shapes
:���������
X
dropout_4/keep_probConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
b
dropout_4/ShapeShapeLeakyRelu_4/Maximum*
out_type0*
T0*
_output_shapes
:
a
dropout_4/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
a
dropout_4/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:���������
�
dropout_4/random_uniform/subSubdropout_4/random_uniform/maxdropout_4/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_4/random_uniform/mulMul&dropout_4/random_uniform/RandomUniformdropout_4/random_uniform/sub*
T0*'
_output_shapes
:���������
�
dropout_4/random_uniformAdddropout_4/random_uniform/muldropout_4/random_uniform/min*
T0*'
_output_shapes
:���������
u
dropout_4/addAdddropout_4/keep_probdropout_4/random_uniform*
T0*'
_output_shapes
:���������
Y
dropout_4/FloorFloordropout_4/add*
T0*'
_output_shapes
:���������
t
dropout_4/divRealDivLeakyRelu_4/Maximumdropout_4/keep_prob*
T0*'
_output_shapes
:���������
f
dropout_4/mulMuldropout_4/divdropout_4/Floor*
T0*'
_output_shapes
:���������
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
~
ArgMaxArgMaxdropout_4/mulArgMax/dimension*
output_type0	*#
_output_shapes
:���������*
T0*

Tidx0
U
one_hot/on_valueConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
V
one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
O
one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
�
one_hotOneHotArgMaxone_hot/depthone_hot/on_valueone_hot/off_value*
axis���������*
T0*'
_output_shapes
:���������*
TI0	
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
{
ArgMax_1ArgMaxlabelsArgMax_1/dimension*
output_type0	*#
_output_shapes
:���������*
T0*

Tidx0
f
loss/absolute_difference/SubSubone_hotlabels*
T0*'
_output_shapes
:���������
s
loss/absolute_difference/AbsAbsloss/absolute_difference/Sub*
T0*'
_output_shapes
:���������
z
5loss/absolute_difference/assert_broadcastable/weightsConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
~
;loss/absolute_difference/assert_broadcastable/weights/shapeConst*
dtype0*
valueB *
_output_shapes
: 
|
:loss/absolute_difference/assert_broadcastable/weights/rankConst*
dtype0*
value	B : *
_output_shapes
: 
�
:loss/absolute_difference/assert_broadcastable/values/shapeShapeloss/absolute_difference/Abs*
out_type0*
T0*
_output_shapes
:
{
9loss/absolute_difference/assert_broadcastable/values/rankConst*
dtype0*
value	B :*
_output_shapes
: 
Q
Iloss/absolute_difference/assert_broadcastable/static_scalar_check_successNoOp
�
$loss/absolute_difference/ToFloat_3/xConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
loss/absolute_difference/MulMulloss/absolute_difference/Abs$loss/absolute_difference/ToFloat_3/x*
T0*'
_output_shapes
:���������
�
loss/absolute_difference/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB"       *
_output_shapes
:
�
loss/absolute_difference/SumSumloss/absolute_difference/Mulloss/absolute_difference/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
,loss/absolute_difference/num_present/Equal/yConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 
�
*loss/absolute_difference/num_present/EqualEqual$loss/absolute_difference/ToFloat_3/x,loss/absolute_difference/num_present/Equal/y*
T0*
_output_shapes
: 
�
?loss/absolute_difference/num_present/zeros_like/shape_as_tensorConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
5loss/absolute_difference/num_present/zeros_like/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 
�
/loss/absolute_difference/num_present/zeros_likeFill?loss/absolute_difference/num_present/zeros_like/shape_as_tensor5loss/absolute_difference/num_present/zeros_like/Const*

index_type0*
T0*
_output_shapes
: 
�
4loss/absolute_difference/num_present/ones_like/ShapeConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
4loss/absolute_difference/num_present/ones_like/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
.loss/absolute_difference/num_present/ones_likeFill4loss/absolute_difference/num_present/ones_like/Shape4loss/absolute_difference/num_present/ones_like/Const*

index_type0*
T0*
_output_shapes
: 
�
+loss/absolute_difference/num_present/SelectSelect*loss/absolute_difference/num_present/Equal/loss/absolute_difference/num_present/zeros_like.loss/absolute_difference/num_present/ones_like*
T0*
_output_shapes
: 
�
Yloss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
Xloss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/weights/rankConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
value	B : *
_output_shapes
: 
�
Xloss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/values/shapeShapeloss/absolute_difference/AbsJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
out_type0*
T0*
_output_shapes
:
�
Wloss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/values/rankConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
value	B :*
_output_shapes
: 
�
gloss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success
�
Floss/absolute_difference/num_present/broadcast_weights/ones_like/ShapeShapeloss/absolute_difference/AbsJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_successh^loss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
T0*
_output_shapes
:
�
Floss/absolute_difference/num_present/broadcast_weights/ones_like/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_successh^loss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
@loss/absolute_difference/num_present/broadcast_weights/ones_likeFillFloss/absolute_difference/num_present/broadcast_weights/ones_like/ShapeFloss/absolute_difference/num_present/broadcast_weights/ones_like/Const*

index_type0*
T0*'
_output_shapes
:���������
�
6loss/absolute_difference/num_present/broadcast_weightsMul+loss/absolute_difference/num_present/Select@loss/absolute_difference/num_present/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
�
*loss/absolute_difference/num_present/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB"       *
_output_shapes
:
�
$loss/absolute_difference/num_presentSum6loss/absolute_difference/num_present/broadcast_weights*loss/absolute_difference/num_present/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
 loss/absolute_difference/Const_1ConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
loss/absolute_difference/Sum_1Sumloss/absolute_difference/Sum loss/absolute_difference/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
"loss/absolute_difference/Greater/yConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 
�
 loss/absolute_difference/GreaterGreater$loss/absolute_difference/num_present"loss/absolute_difference/Greater/y*
T0*
_output_shapes
: 
�
 loss/absolute_difference/Equal/yConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 
�
loss/absolute_difference/EqualEqual$loss/absolute_difference/num_present loss/absolute_difference/Equal/y*
T0*
_output_shapes
: 
�
(loss/absolute_difference/ones_like/ShapeConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
(loss/absolute_difference/ones_like/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
"loss/absolute_difference/ones_likeFill(loss/absolute_difference/ones_like/Shape(loss/absolute_difference/ones_like/Const*

index_type0*
T0*
_output_shapes
: 
�
loss/absolute_difference/SelectSelectloss/absolute_difference/Equal"loss/absolute_difference/ones_like$loss/absolute_difference/num_present*
T0*
_output_shapes
: 
�
loss/absolute_difference/divRealDivloss/absolute_difference/Sum_1loss/absolute_difference/Select*
T0*
_output_shapes
: 
�
3loss/absolute_difference/zeros_like/shape_as_tensorConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
)loss/absolute_difference/zeros_like/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 
�
#loss/absolute_difference/zeros_likeFill3loss/absolute_difference/zeros_like/shape_as_tensor)loss/absolute_difference/zeros_like/Const*

index_type0*
T0*
_output_shapes
: 
�
loss/absolute_difference/valueSelect loss/absolute_difference/Greaterloss/absolute_difference/div#loss/absolute_difference/zeros_like*
T0*
_output_shapes
: 
E
loss/L2LossL2LossVariable/read*
T0*
_output_shapes
: 
O

loss/mul/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
I
loss/mulMul
loss/mul/xloss/L2Loss*
T0*
_output_shapes
: 
O

loss/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
F
loss/addAdd
loss/add/xloss/mul*
T0*
_output_shapes
: 
I
loss/L2Loss_1L2LossVariable_1/read*
T0*
_output_shapes
: 
Q
loss/mul_1/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_1Mulloss/mul_1/xloss/L2Loss_1*
T0*
_output_shapes
: 
H

loss/add_1Addloss/add
loss/mul_1*
T0*
_output_shapes
: 
I
loss/L2Loss_2L2LossVariable_2/read*
T0*
_output_shapes
: 
Q
loss/mul_2/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_2Mulloss/mul_2/xloss/L2Loss_2*
T0*
_output_shapes
: 
J

loss/add_2Add
loss/add_1
loss/mul_2*
T0*
_output_shapes
: 
I
loss/L2Loss_3L2LossVariable_3/read*
T0*
_output_shapes
: 
Q
loss/mul_3/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_3Mulloss/mul_3/xloss/L2Loss_3*
T0*
_output_shapes
: 
J

loss/add_3Add
loss/add_2
loss/mul_3*
T0*
_output_shapes
: 
I
loss/L2Loss_4L2LossVariable_4/read*
T0*
_output_shapes
: 
Q
loss/mul_4/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_4Mulloss/mul_4/xloss/L2Loss_4*
T0*
_output_shapes
: 
J

loss/add_4Add
loss/add_3
loss/mul_4*
T0*
_output_shapes
: 
I
loss/L2Loss_5L2LossVariable_5/read*
T0*
_output_shapes
: 
Q
loss/mul_5/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_5Mulloss/mul_5/xloss/L2Loss_5*
T0*
_output_shapes
: 
J

loss/add_5Add
loss/add_4
loss/mul_5*
T0*
_output_shapes
: 
I
loss/L2Loss_6L2LossVariable_6/read*
T0*
_output_shapes
: 
Q
loss/mul_6/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_6Mulloss/mul_6/xloss/L2Loss_6*
T0*
_output_shapes
: 
J

loss/add_6Add
loss/add_5
loss/mul_6*
T0*
_output_shapes
: 
I
loss/L2Loss_7L2LossVariable_7/read*
T0*
_output_shapes
: 
Q
loss/mul_7/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_7Mulloss/mul_7/xloss/L2Loss_7*
T0*
_output_shapes
: 
J

loss/add_7Add
loss/add_6
loss/mul_7*
T0*
_output_shapes
: 
I
loss/L2Loss_8L2LossVariable_8/read*
T0*
_output_shapes
: 
Q
loss/mul_8/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_8Mulloss/mul_8/xloss/L2Loss_8*
T0*
_output_shapes
: 
J

loss/add_8Add
loss/add_7
loss/mul_8*
T0*
_output_shapes
: 
I
loss/L2Loss_9L2LossVariable_9/read*
T0*
_output_shapes
: 
Q
loss/mul_9/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_9Mulloss/mul_9/xloss/L2Loss_9*
T0*
_output_shapes
: 
J

loss/add_9Add
loss/add_8
loss/mul_9*
T0*
_output_shapes
: 
d
prediction_loss/tagsConst*
dtype0* 
valueB Bprediction_loss*
_output_shapes
: 
w
prediction_lossScalarSummaryprediction_loss/tagsloss/absolute_difference/value*
T0*
_output_shapes
: 
V
reg_loss/tagsConst*
dtype0*
valueB Breg_loss*
_output_shapes
: 
U
reg_lossScalarSummaryreg_loss/tags
loss/add_9*
T0*
_output_shapes
: 
^

loss_1/addAddloss/absolute_difference/value
loss/add_9*
T0*
_output_shapes
: 
Z
total_loss/tagsConst*
dtype0*
valueB B
total_loss*
_output_shapes
: 
Y

total_lossScalarSummarytotal_loss/tags
loss_1/add*
T0*
_output_shapes
: 
n
,accuracy/correct_prediction/ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
�
"accuracy/correct_prediction/ArgMaxArgMaxlabels,accuracy/correct_prediction/ArgMax/dimension*
output_type0	*#
_output_shapes
:���������*
T0*

Tidx0
p
.accuracy/correct_prediction/ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
�
$accuracy/correct_prediction/ArgMax_1ArgMaxone_hot.accuracy/correct_prediction/ArgMax_1/dimension*
output_type0	*#
_output_shapes
:���������*
T0*

Tidx0
�
!accuracy/correct_prediction/EqualEqual"accuracy/correct_prediction/ArgMax$accuracy/correct_prediction/ArgMax_1*
T0	*#
_output_shapes
:���������
~
accuracy/accuracy/CastCast!accuracy/correct_prediction/Equal*

DstT0*

SrcT0
*#
_output_shapes
:���������
a
accuracy/accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
accuracy/accuracy/MeanMeanaccuracy/accuracy/Castaccuracy/accuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Z
accuracy_1/tagsConst*
dtype0*
valueB B
accuracy_1*
_output_shapes
: 
e

accuracy_1ScalarSummaryaccuracy_1/tagsaccuracy/accuracy/Mean*
T0*
_output_shapes
: 
[
Variable_10/initial_valueConst*
dtype0*
value	B : *
_output_shapes
: 
o
Variable_10
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
Variable_10/AssignAssignVariable_10Variable_10/initial_value*
validate_shape(*
_class
loc:@Variable_10*
use_locking(*
T0*
_output_shapes
: 
j
Variable_10/readIdentityVariable_10*
_class
loc:@Variable_10*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
X
gradients/grad_ys_0Const*
dtype0*
valueB
 *  �?*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
T0*
_output_shapes
: 
C
*gradients/loss_1/add_grad/tuple/group_depsNoOp^gradients/Fill
�
2gradients/loss_1/add_grad/tuple/control_dependencyIdentitygradients/Fill+^gradients/loss_1/add_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss_1/add_grad/tuple/control_dependency_1Identitygradients/Fill+^gradients/loss_1/add_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
Hgradients/loss/absolute_difference/value_grad/zeros_like/shape_as_tensorConst*
dtype0*
valueB *
_output_shapes
: 
�
>gradients/loss/absolute_difference/value_grad/zeros_like/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
8gradients/loss/absolute_difference/value_grad/zeros_likeFillHgradients/loss/absolute_difference/value_grad/zeros_like/shape_as_tensor>gradients/loss/absolute_difference/value_grad/zeros_like/Const*

index_type0*
T0*
_output_shapes
: 
�
4gradients/loss/absolute_difference/value_grad/SelectSelect loss/absolute_difference/Greater2gradients/loss_1/add_grad/tuple/control_dependency8gradients/loss/absolute_difference/value_grad/zeros_like*
T0*
_output_shapes
: 
�
6gradients/loss/absolute_difference/value_grad/Select_1Select loss/absolute_difference/Greater8gradients/loss/absolute_difference/value_grad/zeros_like2gradients/loss_1/add_grad/tuple/control_dependency*
T0*
_output_shapes
: 
�
>gradients/loss/absolute_difference/value_grad/tuple/group_depsNoOp5^gradients/loss/absolute_difference/value_grad/Select7^gradients/loss/absolute_difference/value_grad/Select_1
�
Fgradients/loss/absolute_difference/value_grad/tuple/control_dependencyIdentity4gradients/loss/absolute_difference/value_grad/Select?^gradients/loss/absolute_difference/value_grad/tuple/group_deps*G
_class=
;9loc:@gradients/loss/absolute_difference/value_grad/Select*
T0*
_output_shapes
: 
�
Hgradients/loss/absolute_difference/value_grad/tuple/control_dependency_1Identity6gradients/loss/absolute_difference/value_grad/Select_1?^gradients/loss/absolute_difference/value_grad/tuple/group_deps*I
_class?
=;loc:@gradients/loss/absolute_difference/value_grad/Select_1*
T0*
_output_shapes
: 
i
*gradients/loss/add_9_grad/tuple/group_depsNoOp5^gradients/loss_1/add_grad/tuple/control_dependency_1
�
2gradients/loss/add_9_grad/tuple/control_dependencyIdentity4gradients/loss_1/add_grad/tuple/control_dependency_1+^gradients/loss/add_9_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_9_grad/tuple/control_dependency_1Identity4gradients/loss_1/add_grad/tuple/control_dependency_1+^gradients/loss/add_9_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
t
1gradients/loss/absolute_difference/div_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
v
3gradients/loss/absolute_difference/div_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
Agradients/loss/absolute_difference/div_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/loss/absolute_difference/div_grad/Shape3gradients/loss/absolute_difference/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
3gradients/loss/absolute_difference/div_grad/RealDivRealDivFgradients/loss/absolute_difference/value_grad/tuple/control_dependencyloss/absolute_difference/Select*
T0*
_output_shapes
: 
�
/gradients/loss/absolute_difference/div_grad/SumSum3gradients/loss/absolute_difference/div_grad/RealDivAgradients/loss/absolute_difference/div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
3gradients/loss/absolute_difference/div_grad/ReshapeReshape/gradients/loss/absolute_difference/div_grad/Sum1gradients/loss/absolute_difference/div_grad/Shape*
_output_shapes
: *
T0*
Tshape0
w
/gradients/loss/absolute_difference/div_grad/NegNegloss/absolute_difference/Sum_1*
T0*
_output_shapes
: 
�
5gradients/loss/absolute_difference/div_grad/RealDiv_1RealDiv/gradients/loss/absolute_difference/div_grad/Negloss/absolute_difference/Select*
T0*
_output_shapes
: 
�
5gradients/loss/absolute_difference/div_grad/RealDiv_2RealDiv5gradients/loss/absolute_difference/div_grad/RealDiv_1loss/absolute_difference/Select*
T0*
_output_shapes
: 
�
/gradients/loss/absolute_difference/div_grad/mulMulFgradients/loss/absolute_difference/value_grad/tuple/control_dependency5gradients/loss/absolute_difference/div_grad/RealDiv_2*
T0*
_output_shapes
: 
�
1gradients/loss/absolute_difference/div_grad/Sum_1Sum/gradients/loss/absolute_difference/div_grad/mulCgradients/loss/absolute_difference/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
5gradients/loss/absolute_difference/div_grad/Reshape_1Reshape1gradients/loss/absolute_difference/div_grad/Sum_13gradients/loss/absolute_difference/div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
<gradients/loss/absolute_difference/div_grad/tuple/group_depsNoOp4^gradients/loss/absolute_difference/div_grad/Reshape6^gradients/loss/absolute_difference/div_grad/Reshape_1
�
Dgradients/loss/absolute_difference/div_grad/tuple/control_dependencyIdentity3gradients/loss/absolute_difference/div_grad/Reshape=^gradients/loss/absolute_difference/div_grad/tuple/group_deps*F
_class<
:8loc:@gradients/loss/absolute_difference/div_grad/Reshape*
T0*
_output_shapes
: 
�
Fgradients/loss/absolute_difference/div_grad/tuple/control_dependency_1Identity5gradients/loss/absolute_difference/div_grad/Reshape_1=^gradients/loss/absolute_difference/div_grad/tuple/group_deps*H
_class>
<:loc:@gradients/loss/absolute_difference/div_grad/Reshape_1*
T0*
_output_shapes
: 
g
*gradients/loss/add_8_grad/tuple/group_depsNoOp3^gradients/loss/add_9_grad/tuple/control_dependency
�
2gradients/loss/add_8_grad/tuple/control_dependencyIdentity2gradients/loss/add_9_grad/tuple/control_dependency+^gradients/loss/add_8_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_8_grad/tuple/control_dependency_1Identity2gradients/loss/add_9_grad/tuple/control_dependency+^gradients/loss/add_8_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_9_grad/MulMul4gradients/loss/add_9_grad/tuple/control_dependency_1loss/L2Loss_9*
T0*
_output_shapes
: 
�
gradients/loss/mul_9_grad/Mul_1Mul4gradients/loss/add_9_grad/tuple/control_dependency_1loss/mul_9/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_9_grad/tuple/group_depsNoOp^gradients/loss/mul_9_grad/Mul ^gradients/loss/mul_9_grad/Mul_1
�
2gradients/loss/mul_9_grad/tuple/control_dependencyIdentitygradients/loss/mul_9_grad/Mul+^gradients/loss/mul_9_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_9_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_9_grad/tuple/control_dependency_1Identitygradients/loss/mul_9_grad/Mul_1+^gradients/loss/mul_9_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_9_grad/Mul_1*
T0*
_output_shapes
: 
~
;gradients/loss/absolute_difference/Sum_1_grad/Reshape/shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
5gradients/loss/absolute_difference/Sum_1_grad/ReshapeReshapeDgradients/loss/absolute_difference/div_grad/tuple/control_dependency;gradients/loss/absolute_difference/Sum_1_grad/Reshape/shape*
_output_shapes
: *
T0*
Tshape0
v
3gradients/loss/absolute_difference/Sum_1_grad/ConstConst*
dtype0*
valueB *
_output_shapes
: 
�
2gradients/loss/absolute_difference/Sum_1_grad/TileTile5gradients/loss/absolute_difference/Sum_1_grad/Reshape3gradients/loss/absolute_difference/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
�
Igradients/loss/absolute_difference/Select_grad/zeros_like/shape_as_tensorConst*
dtype0*
valueB *
_output_shapes
: 
�
?gradients/loss/absolute_difference/Select_grad/zeros_like/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
9gradients/loss/absolute_difference/Select_grad/zeros_likeFillIgradients/loss/absolute_difference/Select_grad/zeros_like/shape_as_tensor?gradients/loss/absolute_difference/Select_grad/zeros_like/Const*

index_type0*
T0*
_output_shapes
: 
�
5gradients/loss/absolute_difference/Select_grad/SelectSelectloss/absolute_difference/EqualFgradients/loss/absolute_difference/div_grad/tuple/control_dependency_19gradients/loss/absolute_difference/Select_grad/zeros_like*
T0*
_output_shapes
: 
�
7gradients/loss/absolute_difference/Select_grad/Select_1Selectloss/absolute_difference/Equal9gradients/loss/absolute_difference/Select_grad/zeros_likeFgradients/loss/absolute_difference/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
?gradients/loss/absolute_difference/Select_grad/tuple/group_depsNoOp6^gradients/loss/absolute_difference/Select_grad/Select8^gradients/loss/absolute_difference/Select_grad/Select_1
�
Ggradients/loss/absolute_difference/Select_grad/tuple/control_dependencyIdentity5gradients/loss/absolute_difference/Select_grad/Select@^gradients/loss/absolute_difference/Select_grad/tuple/group_deps*H
_class>
<:loc:@gradients/loss/absolute_difference/Select_grad/Select*
T0*
_output_shapes
: 
�
Igradients/loss/absolute_difference/Select_grad/tuple/control_dependency_1Identity7gradients/loss/absolute_difference/Select_grad/Select_1@^gradients/loss/absolute_difference/Select_grad/tuple/group_deps*J
_class@
><loc:@gradients/loss/absolute_difference/Select_grad/Select_1*
T0*
_output_shapes
: 
g
*gradients/loss/add_7_grad/tuple/group_depsNoOp3^gradients/loss/add_8_grad/tuple/control_dependency
�
2gradients/loss/add_7_grad/tuple/control_dependencyIdentity2gradients/loss/add_8_grad/tuple/control_dependency+^gradients/loss/add_7_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_7_grad/tuple/control_dependency_1Identity2gradients/loss/add_8_grad/tuple/control_dependency+^gradients/loss/add_7_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_8_grad/MulMul4gradients/loss/add_8_grad/tuple/control_dependency_1loss/L2Loss_8*
T0*
_output_shapes
: 
�
gradients/loss/mul_8_grad/Mul_1Mul4gradients/loss/add_8_grad/tuple/control_dependency_1loss/mul_8/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_8_grad/tuple/group_depsNoOp^gradients/loss/mul_8_grad/Mul ^gradients/loss/mul_8_grad/Mul_1
�
2gradients/loss/mul_8_grad/tuple/control_dependencyIdentitygradients/loss/mul_8_grad/Mul+^gradients/loss/mul_8_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_8_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_8_grad/tuple/control_dependency_1Identitygradients/loss/mul_8_grad/Mul_1+^gradients/loss/mul_8_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_8_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_9_grad/mulMulVariable_9/read4gradients/loss/mul_9_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
�
9gradients/loss/absolute_difference/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
3gradients/loss/absolute_difference/Sum_grad/ReshapeReshape2gradients/loss/absolute_difference/Sum_1_grad/Tile9gradients/loss/absolute_difference/Sum_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
�
1gradients/loss/absolute_difference/Sum_grad/ShapeShapeloss/absolute_difference/Mul*
out_type0*
T0*
_output_shapes
:
�
0gradients/loss/absolute_difference/Sum_grad/TileTile3gradients/loss/absolute_difference/Sum_grad/Reshape1gradients/loss/absolute_difference/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
g
*gradients/loss/add_6_grad/tuple/group_depsNoOp3^gradients/loss/add_7_grad/tuple/control_dependency
�
2gradients/loss/add_6_grad/tuple/control_dependencyIdentity2gradients/loss/add_7_grad/tuple/control_dependency+^gradients/loss/add_6_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_6_grad/tuple/control_dependency_1Identity2gradients/loss/add_7_grad/tuple/control_dependency+^gradients/loss/add_6_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_7_grad/MulMul4gradients/loss/add_7_grad/tuple/control_dependency_1loss/L2Loss_7*
T0*
_output_shapes
: 
�
gradients/loss/mul_7_grad/Mul_1Mul4gradients/loss/add_7_grad/tuple/control_dependency_1loss/mul_7/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_7_grad/tuple/group_depsNoOp^gradients/loss/mul_7_grad/Mul ^gradients/loss/mul_7_grad/Mul_1
�
2gradients/loss/mul_7_grad/tuple/control_dependencyIdentitygradients/loss/mul_7_grad/Mul+^gradients/loss/mul_7_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_7_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_7_grad/tuple/control_dependency_1Identitygradients/loss/mul_7_grad/Mul_1+^gradients/loss/mul_7_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_7_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_8_grad/mulMulVariable_8/read4gradients/loss/mul_8_grad/tuple/control_dependency_1*
T0*
_output_shapes

:
�
1gradients/loss/absolute_difference/Mul_grad/ShapeShapeloss/absolute_difference/Abs*
out_type0*
T0*
_output_shapes
:
v
3gradients/loss/absolute_difference/Mul_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
Agradients/loss/absolute_difference/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/loss/absolute_difference/Mul_grad/Shape3gradients/loss/absolute_difference/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
/gradients/loss/absolute_difference/Mul_grad/MulMul0gradients/loss/absolute_difference/Sum_grad/Tile$loss/absolute_difference/ToFloat_3/x*
T0*'
_output_shapes
:���������
�
/gradients/loss/absolute_difference/Mul_grad/SumSum/gradients/loss/absolute_difference/Mul_grad/MulAgradients/loss/absolute_difference/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
3gradients/loss/absolute_difference/Mul_grad/ReshapeReshape/gradients/loss/absolute_difference/Mul_grad/Sum1gradients/loss/absolute_difference/Mul_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
1gradients/loss/absolute_difference/Mul_grad/Mul_1Mulloss/absolute_difference/Abs0gradients/loss/absolute_difference/Sum_grad/Tile*
T0*'
_output_shapes
:���������
�
1gradients/loss/absolute_difference/Mul_grad/Sum_1Sum1gradients/loss/absolute_difference/Mul_grad/Mul_1Cgradients/loss/absolute_difference/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
5gradients/loss/absolute_difference/Mul_grad/Reshape_1Reshape1gradients/loss/absolute_difference/Mul_grad/Sum_13gradients/loss/absolute_difference/Mul_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
<gradients/loss/absolute_difference/Mul_grad/tuple/group_depsNoOp4^gradients/loss/absolute_difference/Mul_grad/Reshape6^gradients/loss/absolute_difference/Mul_grad/Reshape_1
�
Dgradients/loss/absolute_difference/Mul_grad/tuple/control_dependencyIdentity3gradients/loss/absolute_difference/Mul_grad/Reshape=^gradients/loss/absolute_difference/Mul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/loss/absolute_difference/Mul_grad/Reshape*
T0*'
_output_shapes
:���������
�
Fgradients/loss/absolute_difference/Mul_grad/tuple/control_dependency_1Identity5gradients/loss/absolute_difference/Mul_grad/Reshape_1=^gradients/loss/absolute_difference/Mul_grad/tuple/group_deps*H
_class>
<:loc:@gradients/loss/absolute_difference/Mul_grad/Reshape_1*
T0*
_output_shapes
: 
�
Agradients/loss/absolute_difference/num_present_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
;gradients/loss/absolute_difference/num_present_grad/ReshapeReshapeIgradients/loss/absolute_difference/Select_grad/tuple/control_dependency_1Agradients/loss/absolute_difference/num_present_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
�
9gradients/loss/absolute_difference/num_present_grad/ShapeShape6loss/absolute_difference/num_present/broadcast_weights*
out_type0*
T0*
_output_shapes
:
�
8gradients/loss/absolute_difference/num_present_grad/TileTile;gradients/loss/absolute_difference/num_present_grad/Reshape9gradients/loss/absolute_difference/num_present_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
g
*gradients/loss/add_5_grad/tuple/group_depsNoOp3^gradients/loss/add_6_grad/tuple/control_dependency
�
2gradients/loss/add_5_grad/tuple/control_dependencyIdentity2gradients/loss/add_6_grad/tuple/control_dependency+^gradients/loss/add_5_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_5_grad/tuple/control_dependency_1Identity2gradients/loss/add_6_grad/tuple/control_dependency+^gradients/loss/add_5_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_6_grad/MulMul4gradients/loss/add_6_grad/tuple/control_dependency_1loss/L2Loss_6*
T0*
_output_shapes
: 
�
gradients/loss/mul_6_grad/Mul_1Mul4gradients/loss/add_6_grad/tuple/control_dependency_1loss/mul_6/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_6_grad/tuple/group_depsNoOp^gradients/loss/mul_6_grad/Mul ^gradients/loss/mul_6_grad/Mul_1
�
2gradients/loss/mul_6_grad/tuple/control_dependencyIdentitygradients/loss/mul_6_grad/Mul+^gradients/loss/mul_6_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_6_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_6_grad/tuple/control_dependency_1Identitygradients/loss/mul_6_grad/Mul_1+^gradients/loss/mul_6_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_6_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_7_grad/mulMulVariable_7/read4gradients/loss/mul_7_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
�
Kgradients/loss/absolute_difference/num_present/broadcast_weights_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
�
Mgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Shape_1Shape@loss/absolute_difference/num_present/broadcast_weights/ones_like*
out_type0*
T0*
_output_shapes
:
�
[gradients/loss/absolute_difference/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsKgradients/loss/absolute_difference/num_present/broadcast_weights_grad/ShapeMgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Igradients/loss/absolute_difference/num_present/broadcast_weights_grad/MulMul8gradients/loss/absolute_difference/num_present_grad/Tile@loss/absolute_difference/num_present/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
�
Igradients/loss/absolute_difference/num_present/broadcast_weights_grad/SumSumIgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Mul[gradients/loss/absolute_difference/num_present/broadcast_weights_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Mgradients/loss/absolute_difference/num_present/broadcast_weights_grad/ReshapeReshapeIgradients/loss/absolute_difference/num_present/broadcast_weights_grad/SumKgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Kgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Mul_1Mul+loss/absolute_difference/num_present/Select8gradients/loss/absolute_difference/num_present_grad/Tile*
T0*'
_output_shapes
:���������
�
Kgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Sum_1SumKgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Mul_1]gradients/loss/absolute_difference/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Ogradients/loss/absolute_difference/num_present/broadcast_weights_grad/Reshape_1ReshapeKgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Sum_1Mgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
Vgradients/loss/absolute_difference/num_present/broadcast_weights_grad/tuple/group_depsNoOpN^gradients/loss/absolute_difference/num_present/broadcast_weights_grad/ReshapeP^gradients/loss/absolute_difference/num_present/broadcast_weights_grad/Reshape_1
�
^gradients/loss/absolute_difference/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityMgradients/loss/absolute_difference/num_present/broadcast_weights_grad/ReshapeW^gradients/loss/absolute_difference/num_present/broadcast_weights_grad/tuple/group_deps*`
_classV
TRloc:@gradients/loss/absolute_difference/num_present/broadcast_weights_grad/Reshape*
T0*
_output_shapes
: 
�
`gradients/loss/absolute_difference/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityOgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Reshape_1W^gradients/loss/absolute_difference/num_present/broadcast_weights_grad/tuple/group_deps*b
_classX
VTloc:@gradients/loss/absolute_difference/num_present/broadcast_weights_grad/Reshape_1*
T0*'
_output_shapes
:���������
g
*gradients/loss/add_4_grad/tuple/group_depsNoOp3^gradients/loss/add_5_grad/tuple/control_dependency
�
2gradients/loss/add_4_grad/tuple/control_dependencyIdentity2gradients/loss/add_5_grad/tuple/control_dependency+^gradients/loss/add_4_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_4_grad/tuple/control_dependency_1Identity2gradients/loss/add_5_grad/tuple/control_dependency+^gradients/loss/add_4_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_5_grad/MulMul4gradients/loss/add_5_grad/tuple/control_dependency_1loss/L2Loss_5*
T0*
_output_shapes
: 
�
gradients/loss/mul_5_grad/Mul_1Mul4gradients/loss/add_5_grad/tuple/control_dependency_1loss/mul_5/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_5_grad/tuple/group_depsNoOp^gradients/loss/mul_5_grad/Mul ^gradients/loss/mul_5_grad/Mul_1
�
2gradients/loss/mul_5_grad/tuple/control_dependencyIdentitygradients/loss/mul_5_grad/Mul+^gradients/loss/mul_5_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_5_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_5_grad/tuple/control_dependency_1Identitygradients/loss/mul_5_grad/Mul_1+^gradients/loss/mul_5_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_5_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_6_grad/mulMulVariable_6/read4gradients/loss/mul_6_grad/tuple/control_dependency_1*
T0*
_output_shapes

:2
�
Ugradients/loss/absolute_difference/num_present/broadcast_weights/ones_like_grad/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
Sgradients/loss/absolute_difference/num_present/broadcast_weights/ones_like_grad/SumSum`gradients/loss/absolute_difference/num_present/broadcast_weights_grad/tuple/control_dependency_1Ugradients/loss/absolute_difference/num_present/broadcast_weights/ones_like_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
g
*gradients/loss/add_3_grad/tuple/group_depsNoOp3^gradients/loss/add_4_grad/tuple/control_dependency
�
2gradients/loss/add_3_grad/tuple/control_dependencyIdentity2gradients/loss/add_4_grad/tuple/control_dependency+^gradients/loss/add_3_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_3_grad/tuple/control_dependency_1Identity2gradients/loss/add_4_grad/tuple/control_dependency+^gradients/loss/add_3_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_4_grad/MulMul4gradients/loss/add_4_grad/tuple/control_dependency_1loss/L2Loss_4*
T0*
_output_shapes
: 
�
gradients/loss/mul_4_grad/Mul_1Mul4gradients/loss/add_4_grad/tuple/control_dependency_1loss/mul_4/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_4_grad/tuple/group_depsNoOp^gradients/loss/mul_4_grad/Mul ^gradients/loss/mul_4_grad/Mul_1
�
2gradients/loss/mul_4_grad/tuple/control_dependencyIdentitygradients/loss/mul_4_grad/Mul+^gradients/loss/mul_4_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_4_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_4_grad/tuple/control_dependency_1Identitygradients/loss/mul_4_grad/Mul_1+^gradients/loss/mul_4_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_4_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_5_grad/mulMulVariable_5/read4gradients/loss/mul_5_grad/tuple/control_dependency_1*
T0*
_output_shapes
:2
g
*gradients/loss/add_2_grad/tuple/group_depsNoOp3^gradients/loss/add_3_grad/tuple/control_dependency
�
2gradients/loss/add_2_grad/tuple/control_dependencyIdentity2gradients/loss/add_3_grad/tuple/control_dependency+^gradients/loss/add_2_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_2_grad/tuple/control_dependency_1Identity2gradients/loss/add_3_grad/tuple/control_dependency+^gradients/loss/add_2_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_3_grad/MulMul4gradients/loss/add_3_grad/tuple/control_dependency_1loss/L2Loss_3*
T0*
_output_shapes
: 
�
gradients/loss/mul_3_grad/Mul_1Mul4gradients/loss/add_3_grad/tuple/control_dependency_1loss/mul_3/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_3_grad/tuple/group_depsNoOp^gradients/loss/mul_3_grad/Mul ^gradients/loss/mul_3_grad/Mul_1
�
2gradients/loss/mul_3_grad/tuple/control_dependencyIdentitygradients/loss/mul_3_grad/Mul+^gradients/loss/mul_3_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_3_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_3_grad/tuple/control_dependency_1Identitygradients/loss/mul_3_grad/Mul_1+^gradients/loss/mul_3_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_3_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_4_grad/mulMulVariable_4/read4gradients/loss/mul_4_grad/tuple/control_dependency_1*
T0*
_output_shapes

:d2
�
0gradients/loss/absolute_difference/Abs_grad/SignSignloss/absolute_difference/Sub*
T0*'
_output_shapes
:���������
�
/gradients/loss/absolute_difference/Abs_grad/mulMulDgradients/loss/absolute_difference/Mul_grad/tuple/control_dependency0gradients/loss/absolute_difference/Abs_grad/Sign*
T0*'
_output_shapes
:���������
g
*gradients/loss/add_1_grad/tuple/group_depsNoOp3^gradients/loss/add_2_grad/tuple/control_dependency
�
2gradients/loss/add_1_grad/tuple/control_dependencyIdentity2gradients/loss/add_2_grad/tuple/control_dependency+^gradients/loss/add_1_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_1_grad/tuple/control_dependency_1Identity2gradients/loss/add_2_grad/tuple/control_dependency+^gradients/loss/add_1_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_2_grad/MulMul4gradients/loss/add_2_grad/tuple/control_dependency_1loss/L2Loss_2*
T0*
_output_shapes
: 
�
gradients/loss/mul_2_grad/Mul_1Mul4gradients/loss/add_2_grad/tuple/control_dependency_1loss/mul_2/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_2_grad/tuple/group_depsNoOp^gradients/loss/mul_2_grad/Mul ^gradients/loss/mul_2_grad/Mul_1
�
2gradients/loss/mul_2_grad/tuple/control_dependencyIdentitygradients/loss/mul_2_grad/Mul+^gradients/loss/mul_2_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_2_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_2_grad/tuple/control_dependency_1Identitygradients/loss/mul_2_grad/Mul_1+^gradients/loss/mul_2_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_2_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_3_grad/mulMulVariable_3/read4gradients/loss/mul_3_grad/tuple/control_dependency_1*
T0*
_output_shapes
:d
x
1gradients/loss/absolute_difference/Sub_grad/ShapeShapeone_hot*
out_type0*
T0*
_output_shapes
:
y
3gradients/loss/absolute_difference/Sub_grad/Shape_1Shapelabels*
out_type0*
T0*
_output_shapes
:
�
Agradients/loss/absolute_difference/Sub_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/loss/absolute_difference/Sub_grad/Shape3gradients/loss/absolute_difference/Sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
/gradients/loss/absolute_difference/Sub_grad/SumSum/gradients/loss/absolute_difference/Abs_grad/mulAgradients/loss/absolute_difference/Sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
3gradients/loss/absolute_difference/Sub_grad/ReshapeReshape/gradients/loss/absolute_difference/Sub_grad/Sum1gradients/loss/absolute_difference/Sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
1gradients/loss/absolute_difference/Sub_grad/Sum_1Sum/gradients/loss/absolute_difference/Abs_grad/mulCgradients/loss/absolute_difference/Sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
/gradients/loss/absolute_difference/Sub_grad/NegNeg1gradients/loss/absolute_difference/Sub_grad/Sum_1*
T0*
_output_shapes
:
�
5gradients/loss/absolute_difference/Sub_grad/Reshape_1Reshape/gradients/loss/absolute_difference/Sub_grad/Neg3gradients/loss/absolute_difference/Sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
<gradients/loss/absolute_difference/Sub_grad/tuple/group_depsNoOp4^gradients/loss/absolute_difference/Sub_grad/Reshape6^gradients/loss/absolute_difference/Sub_grad/Reshape_1
�
Dgradients/loss/absolute_difference/Sub_grad/tuple/control_dependencyIdentity3gradients/loss/absolute_difference/Sub_grad/Reshape=^gradients/loss/absolute_difference/Sub_grad/tuple/group_deps*F
_class<
:8loc:@gradients/loss/absolute_difference/Sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
Fgradients/loss/absolute_difference/Sub_grad/tuple/control_dependency_1Identity5gradients/loss/absolute_difference/Sub_grad/Reshape_1=^gradients/loss/absolute_difference/Sub_grad/tuple/group_deps*H
_class>
<:loc:@gradients/loss/absolute_difference/Sub_grad/Reshape_1*
T0*'
_output_shapes
:���������
e
(gradients/loss/add_grad/tuple/group_depsNoOp3^gradients/loss/add_1_grad/tuple/control_dependency
�
0gradients/loss/add_grad/tuple/control_dependencyIdentity2gradients/loss/add_1_grad/tuple/control_dependency)^gradients/loss/add_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
2gradients/loss/add_grad/tuple/control_dependency_1Identity2gradients/loss/add_1_grad/tuple/control_dependency)^gradients/loss/add_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_1_grad/MulMul4gradients/loss/add_1_grad/tuple/control_dependency_1loss/L2Loss_1*
T0*
_output_shapes
: 
�
gradients/loss/mul_1_grad/Mul_1Mul4gradients/loss/add_1_grad/tuple/control_dependency_1loss/mul_1/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_1_grad/tuple/group_depsNoOp^gradients/loss/mul_1_grad/Mul ^gradients/loss/mul_1_grad/Mul_1
�
2gradients/loss/mul_1_grad/tuple/control_dependencyIdentitygradients/loss/mul_1_grad/Mul+^gradients/loss/mul_1_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_1_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_1_grad/tuple/control_dependency_1Identitygradients/loss/mul_1_grad/Mul_1+^gradients/loss/mul_1_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_1_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_2_grad/mulMulVariable_2/read4gradients/loss/mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	�d
�
gradients/loss/mul_grad/MulMul2gradients/loss/add_grad/tuple/control_dependency_1loss/L2Loss*
T0*
_output_shapes
: 
�
gradients/loss/mul_grad/Mul_1Mul2gradients/loss/add_grad/tuple/control_dependency_1
loss/mul/x*
T0*
_output_shapes
: 
n
(gradients/loss/mul_grad/tuple/group_depsNoOp^gradients/loss/mul_grad/Mul^gradients/loss/mul_grad/Mul_1
�
0gradients/loss/mul_grad/tuple/control_dependencyIdentitygradients/loss/mul_grad/Mul)^gradients/loss/mul_grad/tuple/group_deps*.
_class$
" loc:@gradients/loss/mul_grad/Mul*
T0*
_output_shapes
: 
�
2gradients/loss/mul_grad/tuple/control_dependency_1Identitygradients/loss/mul_grad/Mul_1)^gradients/loss/mul_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_1_grad/mulMulVariable_1/read4gradients/loss/mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
gradients/loss/L2Loss_grad/mulMulVariable/read2gradients/loss/mul_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	�
�
"Variable/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable*
valueB	�*���=*
_output_shapes
:	�
�
Variable/Adagrad
VariableV2*
	container *
_output_shapes
:	�*
dtype0*
shape:	�*
_class
loc:@Variable*
shared_name 
�
Variable/Adagrad/AssignAssignVariable/Adagrad"Variable/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
:	�
z
Variable/Adagrad/readIdentityVariable/Adagrad*
_class
loc:@Variable*
T0*
_output_shapes
:	�
�
$Variable_1/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_1*
valueB�*���=*
_output_shapes	
:�
�
Variable_1/Adagrad
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*
_class
loc:@Variable_1*
shared_name 
�
Variable_1/Adagrad/AssignAssignVariable_1/Adagrad$Variable_1/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes	
:�
|
Variable_1/Adagrad/readIdentityVariable_1/Adagrad*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
�
$Variable_2/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_2*
valueB	�d*���=*
_output_shapes
:	�d
�
Variable_2/Adagrad
VariableV2*
	container *
_output_shapes
:	�d*
dtype0*
shape:	�d*
_class
loc:@Variable_2*
shared_name 
�
Variable_2/Adagrad/AssignAssignVariable_2/Adagrad$Variable_2/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
:	�d
�
Variable_2/Adagrad/readIdentityVariable_2/Adagrad*
_class
loc:@Variable_2*
T0*
_output_shapes
:	�d
�
$Variable_3/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_3*
valueBd*���=*
_output_shapes
:d
�
Variable_3/Adagrad
VariableV2*
	container *
_output_shapes
:d*
dtype0*
shape:d*
_class
loc:@Variable_3*
shared_name 
�
Variable_3/Adagrad/AssignAssignVariable_3/Adagrad$Variable_3/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:d
{
Variable_3/Adagrad/readIdentityVariable_3/Adagrad*
_class
loc:@Variable_3*
T0*
_output_shapes
:d
�
$Variable_4/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_4*
valueBd2*���=*
_output_shapes

:d2
�
Variable_4/Adagrad
VariableV2*
	container *
_output_shapes

:d2*
dtype0*
shape
:d2*
_class
loc:@Variable_4*
shared_name 
�
Variable_4/Adagrad/AssignAssignVariable_4/Adagrad$Variable_4/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:d2

Variable_4/Adagrad/readIdentityVariable_4/Adagrad*
_class
loc:@Variable_4*
T0*
_output_shapes

:d2
�
$Variable_5/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_5*
valueB2*���=*
_output_shapes
:2
�
Variable_5/Adagrad
VariableV2*
	container *
_output_shapes
:2*
dtype0*
shape:2*
_class
loc:@Variable_5*
shared_name 
�
Variable_5/Adagrad/AssignAssignVariable_5/Adagrad$Variable_5/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes
:2
{
Variable_5/Adagrad/readIdentityVariable_5/Adagrad*
_class
loc:@Variable_5*
T0*
_output_shapes
:2
�
$Variable_6/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_6*
valueB2*���=*
_output_shapes

:2
�
Variable_6/Adagrad
VariableV2*
	container *
_output_shapes

:2*
dtype0*
shape
:2*
_class
loc:@Variable_6*
shared_name 
�
Variable_6/Adagrad/AssignAssignVariable_6/Adagrad$Variable_6/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes

:2

Variable_6/Adagrad/readIdentityVariable_6/Adagrad*
_class
loc:@Variable_6*
T0*
_output_shapes

:2
�
$Variable_7/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_7*
valueB*���=*
_output_shapes
:
�
Variable_7/Adagrad
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@Variable_7*
shared_name 
�
Variable_7/Adagrad/AssignAssignVariable_7/Adagrad$Variable_7/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
{
Variable_7/Adagrad/readIdentityVariable_7/Adagrad*
_class
loc:@Variable_7*
T0*
_output_shapes
:
�
$Variable_8/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_8*
valueB*���=*
_output_shapes

:
�
Variable_8/Adagrad
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*
_class
loc:@Variable_8*
shared_name 
�
Variable_8/Adagrad/AssignAssignVariable_8/Adagrad$Variable_8/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_8*
use_locking(*
T0*
_output_shapes

:

Variable_8/Adagrad/readIdentityVariable_8/Adagrad*
_class
loc:@Variable_8*
T0*
_output_shapes

:
�
$Variable_9/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_9*
valueB*���=*
_output_shapes
:
�
Variable_9/Adagrad
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@Variable_9*
shared_name 
�
Variable_9/Adagrad/AssignAssignVariable_9/Adagrad$Variable_9/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_9*
use_locking(*
T0*
_output_shapes
:
{
Variable_9/Adagrad/readIdentityVariable_9/Adagrad*
_class
loc:@Variable_9*
T0*
_output_shapes
:
Z
Adagrad/learning_rateConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
�
$Adagrad/update_Variable/ApplyAdagradApplyAdagradVariableVariable/AdagradAdagrad/learning_rategradients/loss/L2Loss_grad/mul*
_class
loc:@Variable*
use_locking( *
T0*
_output_shapes
:	�
�
&Adagrad/update_Variable_1/ApplyAdagradApplyAdagrad
Variable_1Variable_1/AdagradAdagrad/learning_rate gradients/loss/L2Loss_1_grad/mul*
_class
loc:@Variable_1*
use_locking( *
T0*
_output_shapes	
:�
�
&Adagrad/update_Variable_2/ApplyAdagradApplyAdagrad
Variable_2Variable_2/AdagradAdagrad/learning_rate gradients/loss/L2Loss_2_grad/mul*
_class
loc:@Variable_2*
use_locking( *
T0*
_output_shapes
:	�d
�
&Adagrad/update_Variable_3/ApplyAdagradApplyAdagrad
Variable_3Variable_3/AdagradAdagrad/learning_rate gradients/loss/L2Loss_3_grad/mul*
_class
loc:@Variable_3*
use_locking( *
T0*
_output_shapes
:d
�
&Adagrad/update_Variable_4/ApplyAdagradApplyAdagrad
Variable_4Variable_4/AdagradAdagrad/learning_rate gradients/loss/L2Loss_4_grad/mul*
_class
loc:@Variable_4*
use_locking( *
T0*
_output_shapes

:d2
�
&Adagrad/update_Variable_5/ApplyAdagradApplyAdagrad
Variable_5Variable_5/AdagradAdagrad/learning_rate gradients/loss/L2Loss_5_grad/mul*
_class
loc:@Variable_5*
use_locking( *
T0*
_output_shapes
:2
�
&Adagrad/update_Variable_6/ApplyAdagradApplyAdagrad
Variable_6Variable_6/AdagradAdagrad/learning_rate gradients/loss/L2Loss_6_grad/mul*
_class
loc:@Variable_6*
use_locking( *
T0*
_output_shapes

:2
�
&Adagrad/update_Variable_7/ApplyAdagradApplyAdagrad
Variable_7Variable_7/AdagradAdagrad/learning_rate gradients/loss/L2Loss_7_grad/mul*
_class
loc:@Variable_7*
use_locking( *
T0*
_output_shapes
:
�
&Adagrad/update_Variable_8/ApplyAdagradApplyAdagrad
Variable_8Variable_8/AdagradAdagrad/learning_rate gradients/loss/L2Loss_8_grad/mul*
_class
loc:@Variable_8*
use_locking( *
T0*
_output_shapes

:
�
&Adagrad/update_Variable_9/ApplyAdagradApplyAdagrad
Variable_9Variable_9/AdagradAdagrad/learning_rate gradients/loss/L2Loss_9_grad/mul*
_class
loc:@Variable_9*
use_locking( *
T0*
_output_shapes
:
�
Adagrad/updateNoOp%^Adagrad/update_Variable/ApplyAdagrad'^Adagrad/update_Variable_1/ApplyAdagrad'^Adagrad/update_Variable_2/ApplyAdagrad'^Adagrad/update_Variable_3/ApplyAdagrad'^Adagrad/update_Variable_4/ApplyAdagrad'^Adagrad/update_Variable_5/ApplyAdagrad'^Adagrad/update_Variable_6/ApplyAdagrad'^Adagrad/update_Variable_7/ApplyAdagrad'^Adagrad/update_Variable_8/ApplyAdagrad'^Adagrad/update_Variable_9/ApplyAdagrad
�
Adagrad/valueConst^Adagrad/update*
dtype0*
_class
loc:@Variable_10*
value	B :*
_output_shapes
: 
�
Adagrad	AssignAddVariable_10Adagrad/value*
_class
loc:@Variable_10*
use_locking( *
T0*
_output_shapes
: 
v
Merge/MergeSummaryMergeSummaryprediction_lossreg_loss
total_loss
accuracy_1*
_output_shapes
: *
N"+Ax�gL     N)>	�+�)_��AJژ
��
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
ApplyAdagrad
var"T�
accum"T�
lr"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( 
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
s
	AssignAdd
ref"T�

value"T

output_ref"T�" 
Ttype:
2	"
use_lockingbool( 
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
/
Sign
x"T
y"T"
Ttype:

2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
�
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.7.02v1.7.0-3-g024aecf414��
g
dataPlaceholder*
dtype0*
shape:���������*'
_output_shapes
:���������
i
labelsPlaceholder*
dtype0*
shape:���������*'
_output_shapes
:���������
g
truncated_normal/shapeConst*
dtype0*
valueB"   �   *
_output_shapes
:
Z
truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
\
truncated_normal/stddevConst*
dtype0*
valueB
 *�А=*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:	�
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*
_output_shapes
:	�
n
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes
:	�
~
Variable
VariableV2*
dtype0*
shape:	�*
shared_name *
	container *
_output_shapes
:	�
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
:	�
j
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*
_output_shapes
:	�
`
zeros/shape_as_tensorConst*
dtype0*
valueB:�*
_output_shapes
:
P
zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
i
zerosFillzeros/shape_as_tensorzeros/Const*

index_type0*
T0*
_output_shapes	
:�
x

Variable_1
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
Variable_1/AssignAssign
Variable_1zeros*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
i
truncated_normal_1/shapeConst*
dtype0*
valueB"�   d   *
_output_shapes
:
\
truncated_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_1/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:	�d
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*
_output_shapes
:	�d
t
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*
_output_shapes
:	�d
�

Variable_2
VariableV2*
dtype0*
shape:	�d*
shared_name *
	container *
_output_shapes
:	�d
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
:	�d
p
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*
_output_shapes
:	�d
a
zeros_1/shape_as_tensorConst*
dtype0*
valueB:d*
_output_shapes
:
R
zeros_1/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
n
zeros_1Fillzeros_1/shape_as_tensorzeros_1/Const*

index_type0*
T0*
_output_shapes
:d
v

Variable_3
VariableV2*
dtype0*
shape:d*
shared_name *
	container *
_output_shapes
:d
�
Variable_3/AssignAssign
Variable_3zeros_1*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:d
k
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes
:d
i
truncated_normal_2/shapeConst*
dtype0*
valueB"d   2   *
_output_shapes
:
\
truncated_normal_2/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_2/stddevConst*
dtype0*
valueB
 *��>*
_output_shapes
: 
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes

:d2
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*
_output_shapes

:d2
s
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*
_output_shapes

:d2
~

Variable_4
VariableV2*
dtype0*
shape
:d2*
shared_name *
	container *
_output_shapes

:d2
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:d2
o
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes

:d2
a
zeros_2/shape_as_tensorConst*
dtype0*
valueB:2*
_output_shapes
:
R
zeros_2/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
n
zeros_2Fillzeros_2/shape_as_tensorzeros_2/Const*

index_type0*
T0*
_output_shapes
:2
v

Variable_5
VariableV2*
dtype0*
shape:2*
shared_name *
	container *
_output_shapes
:2
�
Variable_5/AssignAssign
Variable_5zeros_2*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes
:2
k
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes
:2
i
truncated_normal_3/shapeConst*
dtype0*
valueB"2      *
_output_shapes
:
\
truncated_normal_3/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_3/stddevConst*
dtype0*
valueB
 *.�d>*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes

:2
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes

:2
s
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes

:2
~

Variable_6
VariableV2*
dtype0*
shape
:2*
shared_name *
	container *
_output_shapes

:2
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes

:2
o
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes

:2
a
zeros_3/shape_as_tensorConst*
dtype0*
valueB:*
_output_shapes
:
R
zeros_3/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
n
zeros_3Fillzeros_3/shape_as_tensorzeros_3/Const*

index_type0*
T0*
_output_shapes
:
v

Variable_7
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:
�
Variable_7/AssignAssign
Variable_7zeros_3*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:
i
truncated_normal_4/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
\
truncated_normal_4/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_4/stddevConst*
dtype0*
valueB
 *�5?*
_output_shapes
: 
�
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes

:
�
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0*
_output_shapes

:
s
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0*
_output_shapes

:
~

Variable_8
VariableV2*
dtype0*
shape
:*
shared_name *
	container *
_output_shapes

:
�
Variable_8/AssignAssign
Variable_8truncated_normal_4*
validate_shape(*
_class
loc:@Variable_8*
use_locking(*
T0*
_output_shapes

:
o
Variable_8/readIdentity
Variable_8*
_class
loc:@Variable_8*
T0*
_output_shapes

:
a
zeros_4/shape_as_tensorConst*
dtype0*
valueB:*
_output_shapes
:
R
zeros_4/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
n
zeros_4Fillzeros_4/shape_as_tensorzeros_4/Const*

index_type0*
T0*
_output_shapes
:
v

Variable_9
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:
�
Variable_9/AssignAssign
Variable_9zeros_4*
validate_shape(*
_class
loc:@Variable_9*
use_locking(*
T0*
_output_shapes
:
k
Variable_9/readIdentity
Variable_9*
_class
loc:@Variable_9*
T0*
_output_shapes
:
~
MatMulMatMuldataVariable/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:����������
V
addAddMatMulVariable_1/read*
T0*(
_output_shapes
:����������
T
LeakyRelu/alphaConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
]
LeakyRelu/mulMulLeakyRelu/alphaadd*
T0*(
_output_shapes
:����������
c
LeakyRelu/MaximumMaximumLeakyRelu/muladd*
T0*(
_output_shapes
:����������
V
dropout/keep_probConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
^
dropout/ShapeShapeLeakyRelu/Maximum*
out_type0*
T0*
_output_shapes
:
_
dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
_
dropout/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*
seed2 *

seed *
T0*(
_output_shapes
:����������
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 
�
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*(
_output_shapes
:����������
�
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*(
_output_shapes
:����������
p
dropout/addAdddropout/keep_probdropout/random_uniform*
T0*(
_output_shapes
:����������
V
dropout/FloorFloordropout/add*
T0*(
_output_shapes
:����������
o
dropout/divRealDivLeakyRelu/Maximumdropout/keep_prob*
T0*(
_output_shapes
:����������
a
dropout/mulMuldropout/divdropout/Floor*
T0*(
_output_shapes
:����������
�
MatMul_1MatMuldropout/mulVariable_2/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������d
Y
add_1AddMatMul_1Variable_3/read*
T0*'
_output_shapes
:���������d
V
LeakyRelu_1/alphaConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
b
LeakyRelu_1/mulMulLeakyRelu_1/alphaadd_1*
T0*'
_output_shapes
:���������d
h
LeakyRelu_1/MaximumMaximumLeakyRelu_1/muladd_1*
T0*'
_output_shapes
:���������d
X
dropout_1/keep_probConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
b
dropout_1/ShapeShapeLeakyRelu_1/Maximum*
out_type0*
T0*
_output_shapes
:
a
dropout_1/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
a
dropout_1/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:���������d
�
dropout_1/random_uniform/subSubdropout_1/random_uniform/maxdropout_1/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_1/random_uniform/mulMul&dropout_1/random_uniform/RandomUniformdropout_1/random_uniform/sub*
T0*'
_output_shapes
:���������d
�
dropout_1/random_uniformAdddropout_1/random_uniform/muldropout_1/random_uniform/min*
T0*'
_output_shapes
:���������d
u
dropout_1/addAdddropout_1/keep_probdropout_1/random_uniform*
T0*'
_output_shapes
:���������d
Y
dropout_1/FloorFloordropout_1/add*
T0*'
_output_shapes
:���������d
t
dropout_1/divRealDivLeakyRelu_1/Maximumdropout_1/keep_prob*
T0*'
_output_shapes
:���������d
f
dropout_1/mulMuldropout_1/divdropout_1/Floor*
T0*'
_output_shapes
:���������d
�
MatMul_2MatMuldropout_1/mulVariable_4/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������2
Y
add_2AddMatMul_2Variable_5/read*
T0*'
_output_shapes
:���������2
V
LeakyRelu_2/alphaConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
b
LeakyRelu_2/mulMulLeakyRelu_2/alphaadd_2*
T0*'
_output_shapes
:���������2
h
LeakyRelu_2/MaximumMaximumLeakyRelu_2/muladd_2*
T0*'
_output_shapes
:���������2
X
dropout_2/keep_probConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
b
dropout_2/ShapeShapeLeakyRelu_2/Maximum*
out_type0*
T0*
_output_shapes
:
a
dropout_2/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
a
dropout_2/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:���������2
�
dropout_2/random_uniform/subSubdropout_2/random_uniform/maxdropout_2/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_2/random_uniform/mulMul&dropout_2/random_uniform/RandomUniformdropout_2/random_uniform/sub*
T0*'
_output_shapes
:���������2
�
dropout_2/random_uniformAdddropout_2/random_uniform/muldropout_2/random_uniform/min*
T0*'
_output_shapes
:���������2
u
dropout_2/addAdddropout_2/keep_probdropout_2/random_uniform*
T0*'
_output_shapes
:���������2
Y
dropout_2/FloorFloordropout_2/add*
T0*'
_output_shapes
:���������2
t
dropout_2/divRealDivLeakyRelu_2/Maximumdropout_2/keep_prob*
T0*'
_output_shapes
:���������2
f
dropout_2/mulMuldropout_2/divdropout_2/Floor*
T0*'
_output_shapes
:���������2
�
MatMul_3MatMuldropout_2/mulVariable_6/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
Y
add_3AddMatMul_3Variable_7/read*
T0*'
_output_shapes
:���������
V
LeakyRelu_3/alphaConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
b
LeakyRelu_3/mulMulLeakyRelu_3/alphaadd_3*
T0*'
_output_shapes
:���������
h
LeakyRelu_3/MaximumMaximumLeakyRelu_3/muladd_3*
T0*'
_output_shapes
:���������
X
dropout_3/keep_probConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
b
dropout_3/ShapeShapeLeakyRelu_3/Maximum*
out_type0*
T0*
_output_shapes
:
a
dropout_3/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
a
dropout_3/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:���������
�
dropout_3/random_uniform/subSubdropout_3/random_uniform/maxdropout_3/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_3/random_uniform/mulMul&dropout_3/random_uniform/RandomUniformdropout_3/random_uniform/sub*
T0*'
_output_shapes
:���������
�
dropout_3/random_uniformAdddropout_3/random_uniform/muldropout_3/random_uniform/min*
T0*'
_output_shapes
:���������
u
dropout_3/addAdddropout_3/keep_probdropout_3/random_uniform*
T0*'
_output_shapes
:���������
Y
dropout_3/FloorFloordropout_3/add*
T0*'
_output_shapes
:���������
t
dropout_3/divRealDivLeakyRelu_3/Maximumdropout_3/keep_prob*
T0*'
_output_shapes
:���������
f
dropout_3/mulMuldropout_3/divdropout_3/Floor*
T0*'
_output_shapes
:���������
�
MatMul_4MatMuldropout_3/mulVariable_8/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
Y
add_4AddMatMul_4Variable_9/read*
T0*'
_output_shapes
:���������
V
LeakyRelu_4/alphaConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
b
LeakyRelu_4/mulMulLeakyRelu_4/alphaadd_4*
T0*'
_output_shapes
:���������
h
LeakyRelu_4/MaximumMaximumLeakyRelu_4/muladd_4*
T0*'
_output_shapes
:���������
X
dropout_4/keep_probConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
b
dropout_4/ShapeShapeLeakyRelu_4/Maximum*
out_type0*
T0*
_output_shapes
:
a
dropout_4/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
a
dropout_4/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:���������
�
dropout_4/random_uniform/subSubdropout_4/random_uniform/maxdropout_4/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_4/random_uniform/mulMul&dropout_4/random_uniform/RandomUniformdropout_4/random_uniform/sub*
T0*'
_output_shapes
:���������
�
dropout_4/random_uniformAdddropout_4/random_uniform/muldropout_4/random_uniform/min*
T0*'
_output_shapes
:���������
u
dropout_4/addAdddropout_4/keep_probdropout_4/random_uniform*
T0*'
_output_shapes
:���������
Y
dropout_4/FloorFloordropout_4/add*
T0*'
_output_shapes
:���������
t
dropout_4/divRealDivLeakyRelu_4/Maximumdropout_4/keep_prob*
T0*'
_output_shapes
:���������
f
dropout_4/mulMuldropout_4/divdropout_4/Floor*
T0*'
_output_shapes
:���������
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
~
ArgMaxArgMaxdropout_4/mulArgMax/dimension*
output_type0	*

Tidx0*
T0*#
_output_shapes
:���������
U
one_hot/on_valueConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
V
one_hot/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
O
one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
�
one_hotOneHotArgMaxone_hot/depthone_hot/on_valueone_hot/off_value*
TI0	*'
_output_shapes
:���������*
T0*
axis���������
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
{
ArgMax_1ArgMaxlabelsArgMax_1/dimension*
output_type0	*

Tidx0*
T0*#
_output_shapes
:���������
f
loss/absolute_difference/SubSubone_hotlabels*
T0*'
_output_shapes
:���������
s
loss/absolute_difference/AbsAbsloss/absolute_difference/Sub*
T0*'
_output_shapes
:���������
z
5loss/absolute_difference/assert_broadcastable/weightsConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
~
;loss/absolute_difference/assert_broadcastable/weights/shapeConst*
dtype0*
valueB *
_output_shapes
: 
|
:loss/absolute_difference/assert_broadcastable/weights/rankConst*
dtype0*
value	B : *
_output_shapes
: 
�
:loss/absolute_difference/assert_broadcastable/values/shapeShapeloss/absolute_difference/Abs*
out_type0*
T0*
_output_shapes
:
{
9loss/absolute_difference/assert_broadcastable/values/rankConst*
dtype0*
value	B :*
_output_shapes
: 
Q
Iloss/absolute_difference/assert_broadcastable/static_scalar_check_successNoOp
�
$loss/absolute_difference/ToFloat_3/xConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
loss/absolute_difference/MulMulloss/absolute_difference/Abs$loss/absolute_difference/ToFloat_3/x*
T0*'
_output_shapes
:���������
�
loss/absolute_difference/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB"       *
_output_shapes
:
�
loss/absolute_difference/SumSumloss/absolute_difference/Mulloss/absolute_difference/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
,loss/absolute_difference/num_present/Equal/yConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 
�
*loss/absolute_difference/num_present/EqualEqual$loss/absolute_difference/ToFloat_3/x,loss/absolute_difference/num_present/Equal/y*
T0*
_output_shapes
: 
�
?loss/absolute_difference/num_present/zeros_like/shape_as_tensorConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
5loss/absolute_difference/num_present/zeros_like/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 
�
/loss/absolute_difference/num_present/zeros_likeFill?loss/absolute_difference/num_present/zeros_like/shape_as_tensor5loss/absolute_difference/num_present/zeros_like/Const*

index_type0*
T0*
_output_shapes
: 
�
4loss/absolute_difference/num_present/ones_like/ShapeConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
4loss/absolute_difference/num_present/ones_like/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
.loss/absolute_difference/num_present/ones_likeFill4loss/absolute_difference/num_present/ones_like/Shape4loss/absolute_difference/num_present/ones_like/Const*

index_type0*
T0*
_output_shapes
: 
�
+loss/absolute_difference/num_present/SelectSelect*loss/absolute_difference/num_present/Equal/loss/absolute_difference/num_present/zeros_like.loss/absolute_difference/num_present/ones_like*
T0*
_output_shapes
: 
�
Yloss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
Xloss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/weights/rankConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
value	B : *
_output_shapes
: 
�
Xloss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/values/shapeShapeloss/absolute_difference/AbsJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
out_type0*
T0*
_output_shapes
:
�
Wloss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/values/rankConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
value	B :*
_output_shapes
: 
�
gloss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success
�
Floss/absolute_difference/num_present/broadcast_weights/ones_like/ShapeShapeloss/absolute_difference/AbsJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_successh^loss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
out_type0*
T0*
_output_shapes
:
�
Floss/absolute_difference/num_present/broadcast_weights/ones_like/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_successh^loss/absolute_difference/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
@loss/absolute_difference/num_present/broadcast_weights/ones_likeFillFloss/absolute_difference/num_present/broadcast_weights/ones_like/ShapeFloss/absolute_difference/num_present/broadcast_weights/ones_like/Const*

index_type0*
T0*'
_output_shapes
:���������
�
6loss/absolute_difference/num_present/broadcast_weightsMul+loss/absolute_difference/num_present/Select@loss/absolute_difference/num_present/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
�
*loss/absolute_difference/num_present/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB"       *
_output_shapes
:
�
$loss/absolute_difference/num_presentSum6loss/absolute_difference/num_present/broadcast_weights*loss/absolute_difference/num_present/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
 loss/absolute_difference/Const_1ConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
loss/absolute_difference/Sum_1Sumloss/absolute_difference/Sum loss/absolute_difference/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
"loss/absolute_difference/Greater/yConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 
�
 loss/absolute_difference/GreaterGreater$loss/absolute_difference/num_present"loss/absolute_difference/Greater/y*
T0*
_output_shapes
: 
�
 loss/absolute_difference/Equal/yConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 
�
loss/absolute_difference/EqualEqual$loss/absolute_difference/num_present loss/absolute_difference/Equal/y*
T0*
_output_shapes
: 
�
(loss/absolute_difference/ones_like/ShapeConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
(loss/absolute_difference/ones_like/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
"loss/absolute_difference/ones_likeFill(loss/absolute_difference/ones_like/Shape(loss/absolute_difference/ones_like/Const*

index_type0*
T0*
_output_shapes
: 
�
loss/absolute_difference/SelectSelectloss/absolute_difference/Equal"loss/absolute_difference/ones_like$loss/absolute_difference/num_present*
T0*
_output_shapes
: 
�
loss/absolute_difference/divRealDivloss/absolute_difference/Sum_1loss/absolute_difference/Select*
T0*
_output_shapes
: 
�
3loss/absolute_difference/zeros_like/shape_as_tensorConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
)loss/absolute_difference/zeros_like/ConstConstJ^loss/absolute_difference/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 
�
#loss/absolute_difference/zeros_likeFill3loss/absolute_difference/zeros_like/shape_as_tensor)loss/absolute_difference/zeros_like/Const*

index_type0*
T0*
_output_shapes
: 
�
loss/absolute_difference/valueSelect loss/absolute_difference/Greaterloss/absolute_difference/div#loss/absolute_difference/zeros_like*
T0*
_output_shapes
: 
E
loss/L2LossL2LossVariable/read*
T0*
_output_shapes
: 
O

loss/mul/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
I
loss/mulMul
loss/mul/xloss/L2Loss*
T0*
_output_shapes
: 
O

loss/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
F
loss/addAdd
loss/add/xloss/mul*
T0*
_output_shapes
: 
I
loss/L2Loss_1L2LossVariable_1/read*
T0*
_output_shapes
: 
Q
loss/mul_1/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_1Mulloss/mul_1/xloss/L2Loss_1*
T0*
_output_shapes
: 
H

loss/add_1Addloss/add
loss/mul_1*
T0*
_output_shapes
: 
I
loss/L2Loss_2L2LossVariable_2/read*
T0*
_output_shapes
: 
Q
loss/mul_2/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_2Mulloss/mul_2/xloss/L2Loss_2*
T0*
_output_shapes
: 
J

loss/add_2Add
loss/add_1
loss/mul_2*
T0*
_output_shapes
: 
I
loss/L2Loss_3L2LossVariable_3/read*
T0*
_output_shapes
: 
Q
loss/mul_3/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_3Mulloss/mul_3/xloss/L2Loss_3*
T0*
_output_shapes
: 
J

loss/add_3Add
loss/add_2
loss/mul_3*
T0*
_output_shapes
: 
I
loss/L2Loss_4L2LossVariable_4/read*
T0*
_output_shapes
: 
Q
loss/mul_4/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_4Mulloss/mul_4/xloss/L2Loss_4*
T0*
_output_shapes
: 
J

loss/add_4Add
loss/add_3
loss/mul_4*
T0*
_output_shapes
: 
I
loss/L2Loss_5L2LossVariable_5/read*
T0*
_output_shapes
: 
Q
loss/mul_5/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_5Mulloss/mul_5/xloss/L2Loss_5*
T0*
_output_shapes
: 
J

loss/add_5Add
loss/add_4
loss/mul_5*
T0*
_output_shapes
: 
I
loss/L2Loss_6L2LossVariable_6/read*
T0*
_output_shapes
: 
Q
loss/mul_6/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_6Mulloss/mul_6/xloss/L2Loss_6*
T0*
_output_shapes
: 
J

loss/add_6Add
loss/add_5
loss/mul_6*
T0*
_output_shapes
: 
I
loss/L2Loss_7L2LossVariable_7/read*
T0*
_output_shapes
: 
Q
loss/mul_7/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_7Mulloss/mul_7/xloss/L2Loss_7*
T0*
_output_shapes
: 
J

loss/add_7Add
loss/add_6
loss/mul_7*
T0*
_output_shapes
: 
I
loss/L2Loss_8L2LossVariable_8/read*
T0*
_output_shapes
: 
Q
loss/mul_8/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_8Mulloss/mul_8/xloss/L2Loss_8*
T0*
_output_shapes
: 
J

loss/add_8Add
loss/add_7
loss/mul_8*
T0*
_output_shapes
: 
I
loss/L2Loss_9L2LossVariable_9/read*
T0*
_output_shapes
: 
Q
loss/mul_9/xConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

loss/mul_9Mulloss/mul_9/xloss/L2Loss_9*
T0*
_output_shapes
: 
J

loss/add_9Add
loss/add_8
loss/mul_9*
T0*
_output_shapes
: 
d
prediction_loss/tagsConst*
dtype0* 
valueB Bprediction_loss*
_output_shapes
: 
w
prediction_lossScalarSummaryprediction_loss/tagsloss/absolute_difference/value*
T0*
_output_shapes
: 
V
reg_loss/tagsConst*
dtype0*
valueB Breg_loss*
_output_shapes
: 
U
reg_lossScalarSummaryreg_loss/tags
loss/add_9*
T0*
_output_shapes
: 
^

loss_1/addAddloss/absolute_difference/value
loss/add_9*
T0*
_output_shapes
: 
Z
total_loss/tagsConst*
dtype0*
valueB B
total_loss*
_output_shapes
: 
Y

total_lossScalarSummarytotal_loss/tags
loss_1/add*
T0*
_output_shapes
: 
n
,accuracy/correct_prediction/ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
�
"accuracy/correct_prediction/ArgMaxArgMaxlabels,accuracy/correct_prediction/ArgMax/dimension*
output_type0	*

Tidx0*
T0*#
_output_shapes
:���������
p
.accuracy/correct_prediction/ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
�
$accuracy/correct_prediction/ArgMax_1ArgMaxone_hot.accuracy/correct_prediction/ArgMax_1/dimension*
output_type0	*

Tidx0*
T0*#
_output_shapes
:���������
�
!accuracy/correct_prediction/EqualEqual"accuracy/correct_prediction/ArgMax$accuracy/correct_prediction/ArgMax_1*
T0	*#
_output_shapes
:���������
~
accuracy/accuracy/CastCast!accuracy/correct_prediction/Equal*

DstT0*

SrcT0
*#
_output_shapes
:���������
a
accuracy/accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
accuracy/accuracy/MeanMeanaccuracy/accuracy/Castaccuracy/accuracy/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Z
accuracy_1/tagsConst*
dtype0*
valueB B
accuracy_1*
_output_shapes
: 
e

accuracy_1ScalarSummaryaccuracy_1/tagsaccuracy/accuracy/Mean*
T0*
_output_shapes
: 
[
Variable_10/initial_valueConst*
dtype0*
value	B : *
_output_shapes
: 
o
Variable_10
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
Variable_10/AssignAssignVariable_10Variable_10/initial_value*
validate_shape(*
_class
loc:@Variable_10*
use_locking(*
T0*
_output_shapes
: 
j
Variable_10/readIdentityVariable_10*
_class
loc:@Variable_10*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
X
gradients/grad_ys_0Const*
dtype0*
valueB
 *  �?*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
T0*
_output_shapes
: 
C
*gradients/loss_1/add_grad/tuple/group_depsNoOp^gradients/Fill
�
2gradients/loss_1/add_grad/tuple/control_dependencyIdentitygradients/Fill+^gradients/loss_1/add_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss_1/add_grad/tuple/control_dependency_1Identitygradients/Fill+^gradients/loss_1/add_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
Hgradients/loss/absolute_difference/value_grad/zeros_like/shape_as_tensorConst*
dtype0*
valueB *
_output_shapes
: 
�
>gradients/loss/absolute_difference/value_grad/zeros_like/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
8gradients/loss/absolute_difference/value_grad/zeros_likeFillHgradients/loss/absolute_difference/value_grad/zeros_like/shape_as_tensor>gradients/loss/absolute_difference/value_grad/zeros_like/Const*

index_type0*
T0*
_output_shapes
: 
�
4gradients/loss/absolute_difference/value_grad/SelectSelect loss/absolute_difference/Greater2gradients/loss_1/add_grad/tuple/control_dependency8gradients/loss/absolute_difference/value_grad/zeros_like*
T0*
_output_shapes
: 
�
6gradients/loss/absolute_difference/value_grad/Select_1Select loss/absolute_difference/Greater8gradients/loss/absolute_difference/value_grad/zeros_like2gradients/loss_1/add_grad/tuple/control_dependency*
T0*
_output_shapes
: 
�
>gradients/loss/absolute_difference/value_grad/tuple/group_depsNoOp5^gradients/loss/absolute_difference/value_grad/Select7^gradients/loss/absolute_difference/value_grad/Select_1
�
Fgradients/loss/absolute_difference/value_grad/tuple/control_dependencyIdentity4gradients/loss/absolute_difference/value_grad/Select?^gradients/loss/absolute_difference/value_grad/tuple/group_deps*G
_class=
;9loc:@gradients/loss/absolute_difference/value_grad/Select*
T0*
_output_shapes
: 
�
Hgradients/loss/absolute_difference/value_grad/tuple/control_dependency_1Identity6gradients/loss/absolute_difference/value_grad/Select_1?^gradients/loss/absolute_difference/value_grad/tuple/group_deps*I
_class?
=;loc:@gradients/loss/absolute_difference/value_grad/Select_1*
T0*
_output_shapes
: 
i
*gradients/loss/add_9_grad/tuple/group_depsNoOp5^gradients/loss_1/add_grad/tuple/control_dependency_1
�
2gradients/loss/add_9_grad/tuple/control_dependencyIdentity4gradients/loss_1/add_grad/tuple/control_dependency_1+^gradients/loss/add_9_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_9_grad/tuple/control_dependency_1Identity4gradients/loss_1/add_grad/tuple/control_dependency_1+^gradients/loss/add_9_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
t
1gradients/loss/absolute_difference/div_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
v
3gradients/loss/absolute_difference/div_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
Agradients/loss/absolute_difference/div_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/loss/absolute_difference/div_grad/Shape3gradients/loss/absolute_difference/div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
3gradients/loss/absolute_difference/div_grad/RealDivRealDivFgradients/loss/absolute_difference/value_grad/tuple/control_dependencyloss/absolute_difference/Select*
T0*
_output_shapes
: 
�
/gradients/loss/absolute_difference/div_grad/SumSum3gradients/loss/absolute_difference/div_grad/RealDivAgradients/loss/absolute_difference/div_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
3gradients/loss/absolute_difference/div_grad/ReshapeReshape/gradients/loss/absolute_difference/div_grad/Sum1gradients/loss/absolute_difference/div_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
w
/gradients/loss/absolute_difference/div_grad/NegNegloss/absolute_difference/Sum_1*
T0*
_output_shapes
: 
�
5gradients/loss/absolute_difference/div_grad/RealDiv_1RealDiv/gradients/loss/absolute_difference/div_grad/Negloss/absolute_difference/Select*
T0*
_output_shapes
: 
�
5gradients/loss/absolute_difference/div_grad/RealDiv_2RealDiv5gradients/loss/absolute_difference/div_grad/RealDiv_1loss/absolute_difference/Select*
T0*
_output_shapes
: 
�
/gradients/loss/absolute_difference/div_grad/mulMulFgradients/loss/absolute_difference/value_grad/tuple/control_dependency5gradients/loss/absolute_difference/div_grad/RealDiv_2*
T0*
_output_shapes
: 
�
1gradients/loss/absolute_difference/div_grad/Sum_1Sum/gradients/loss/absolute_difference/div_grad/mulCgradients/loss/absolute_difference/div_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
5gradients/loss/absolute_difference/div_grad/Reshape_1Reshape1gradients/loss/absolute_difference/div_grad/Sum_13gradients/loss/absolute_difference/div_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
�
<gradients/loss/absolute_difference/div_grad/tuple/group_depsNoOp4^gradients/loss/absolute_difference/div_grad/Reshape6^gradients/loss/absolute_difference/div_grad/Reshape_1
�
Dgradients/loss/absolute_difference/div_grad/tuple/control_dependencyIdentity3gradients/loss/absolute_difference/div_grad/Reshape=^gradients/loss/absolute_difference/div_grad/tuple/group_deps*F
_class<
:8loc:@gradients/loss/absolute_difference/div_grad/Reshape*
T0*
_output_shapes
: 
�
Fgradients/loss/absolute_difference/div_grad/tuple/control_dependency_1Identity5gradients/loss/absolute_difference/div_grad/Reshape_1=^gradients/loss/absolute_difference/div_grad/tuple/group_deps*H
_class>
<:loc:@gradients/loss/absolute_difference/div_grad/Reshape_1*
T0*
_output_shapes
: 
g
*gradients/loss/add_8_grad/tuple/group_depsNoOp3^gradients/loss/add_9_grad/tuple/control_dependency
�
2gradients/loss/add_8_grad/tuple/control_dependencyIdentity2gradients/loss/add_9_grad/tuple/control_dependency+^gradients/loss/add_8_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_8_grad/tuple/control_dependency_1Identity2gradients/loss/add_9_grad/tuple/control_dependency+^gradients/loss/add_8_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_9_grad/MulMul4gradients/loss/add_9_grad/tuple/control_dependency_1loss/L2Loss_9*
T0*
_output_shapes
: 
�
gradients/loss/mul_9_grad/Mul_1Mul4gradients/loss/add_9_grad/tuple/control_dependency_1loss/mul_9/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_9_grad/tuple/group_depsNoOp^gradients/loss/mul_9_grad/Mul ^gradients/loss/mul_9_grad/Mul_1
�
2gradients/loss/mul_9_grad/tuple/control_dependencyIdentitygradients/loss/mul_9_grad/Mul+^gradients/loss/mul_9_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_9_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_9_grad/tuple/control_dependency_1Identitygradients/loss/mul_9_grad/Mul_1+^gradients/loss/mul_9_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_9_grad/Mul_1*
T0*
_output_shapes
: 
~
;gradients/loss/absolute_difference/Sum_1_grad/Reshape/shapeConst*
dtype0*
valueB *
_output_shapes
: 
�
5gradients/loss/absolute_difference/Sum_1_grad/ReshapeReshapeDgradients/loss/absolute_difference/div_grad/tuple/control_dependency;gradients/loss/absolute_difference/Sum_1_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
: 
v
3gradients/loss/absolute_difference/Sum_1_grad/ConstConst*
dtype0*
valueB *
_output_shapes
: 
�
2gradients/loss/absolute_difference/Sum_1_grad/TileTile5gradients/loss/absolute_difference/Sum_1_grad/Reshape3gradients/loss/absolute_difference/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
�
Igradients/loss/absolute_difference/Select_grad/zeros_like/shape_as_tensorConst*
dtype0*
valueB *
_output_shapes
: 
�
?gradients/loss/absolute_difference/Select_grad/zeros_like/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
9gradients/loss/absolute_difference/Select_grad/zeros_likeFillIgradients/loss/absolute_difference/Select_grad/zeros_like/shape_as_tensor?gradients/loss/absolute_difference/Select_grad/zeros_like/Const*

index_type0*
T0*
_output_shapes
: 
�
5gradients/loss/absolute_difference/Select_grad/SelectSelectloss/absolute_difference/EqualFgradients/loss/absolute_difference/div_grad/tuple/control_dependency_19gradients/loss/absolute_difference/Select_grad/zeros_like*
T0*
_output_shapes
: 
�
7gradients/loss/absolute_difference/Select_grad/Select_1Selectloss/absolute_difference/Equal9gradients/loss/absolute_difference/Select_grad/zeros_likeFgradients/loss/absolute_difference/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
�
?gradients/loss/absolute_difference/Select_grad/tuple/group_depsNoOp6^gradients/loss/absolute_difference/Select_grad/Select8^gradients/loss/absolute_difference/Select_grad/Select_1
�
Ggradients/loss/absolute_difference/Select_grad/tuple/control_dependencyIdentity5gradients/loss/absolute_difference/Select_grad/Select@^gradients/loss/absolute_difference/Select_grad/tuple/group_deps*H
_class>
<:loc:@gradients/loss/absolute_difference/Select_grad/Select*
T0*
_output_shapes
: 
�
Igradients/loss/absolute_difference/Select_grad/tuple/control_dependency_1Identity7gradients/loss/absolute_difference/Select_grad/Select_1@^gradients/loss/absolute_difference/Select_grad/tuple/group_deps*J
_class@
><loc:@gradients/loss/absolute_difference/Select_grad/Select_1*
T0*
_output_shapes
: 
g
*gradients/loss/add_7_grad/tuple/group_depsNoOp3^gradients/loss/add_8_grad/tuple/control_dependency
�
2gradients/loss/add_7_grad/tuple/control_dependencyIdentity2gradients/loss/add_8_grad/tuple/control_dependency+^gradients/loss/add_7_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_7_grad/tuple/control_dependency_1Identity2gradients/loss/add_8_grad/tuple/control_dependency+^gradients/loss/add_7_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_8_grad/MulMul4gradients/loss/add_8_grad/tuple/control_dependency_1loss/L2Loss_8*
T0*
_output_shapes
: 
�
gradients/loss/mul_8_grad/Mul_1Mul4gradients/loss/add_8_grad/tuple/control_dependency_1loss/mul_8/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_8_grad/tuple/group_depsNoOp^gradients/loss/mul_8_grad/Mul ^gradients/loss/mul_8_grad/Mul_1
�
2gradients/loss/mul_8_grad/tuple/control_dependencyIdentitygradients/loss/mul_8_grad/Mul+^gradients/loss/mul_8_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_8_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_8_grad/tuple/control_dependency_1Identitygradients/loss/mul_8_grad/Mul_1+^gradients/loss/mul_8_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_8_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_9_grad/mulMulVariable_9/read4gradients/loss/mul_9_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
�
9gradients/loss/absolute_difference/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
3gradients/loss/absolute_difference/Sum_grad/ReshapeReshape2gradients/loss/absolute_difference/Sum_1_grad/Tile9gradients/loss/absolute_difference/Sum_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes

:
�
1gradients/loss/absolute_difference/Sum_grad/ShapeShapeloss/absolute_difference/Mul*
out_type0*
T0*
_output_shapes
:
�
0gradients/loss/absolute_difference/Sum_grad/TileTile3gradients/loss/absolute_difference/Sum_grad/Reshape1gradients/loss/absolute_difference/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
g
*gradients/loss/add_6_grad/tuple/group_depsNoOp3^gradients/loss/add_7_grad/tuple/control_dependency
�
2gradients/loss/add_6_grad/tuple/control_dependencyIdentity2gradients/loss/add_7_grad/tuple/control_dependency+^gradients/loss/add_6_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_6_grad/tuple/control_dependency_1Identity2gradients/loss/add_7_grad/tuple/control_dependency+^gradients/loss/add_6_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_7_grad/MulMul4gradients/loss/add_7_grad/tuple/control_dependency_1loss/L2Loss_7*
T0*
_output_shapes
: 
�
gradients/loss/mul_7_grad/Mul_1Mul4gradients/loss/add_7_grad/tuple/control_dependency_1loss/mul_7/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_7_grad/tuple/group_depsNoOp^gradients/loss/mul_7_grad/Mul ^gradients/loss/mul_7_grad/Mul_1
�
2gradients/loss/mul_7_grad/tuple/control_dependencyIdentitygradients/loss/mul_7_grad/Mul+^gradients/loss/mul_7_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_7_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_7_grad/tuple/control_dependency_1Identitygradients/loss/mul_7_grad/Mul_1+^gradients/loss/mul_7_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_7_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_8_grad/mulMulVariable_8/read4gradients/loss/mul_8_grad/tuple/control_dependency_1*
T0*
_output_shapes

:
�
1gradients/loss/absolute_difference/Mul_grad/ShapeShapeloss/absolute_difference/Abs*
out_type0*
T0*
_output_shapes
:
v
3gradients/loss/absolute_difference/Mul_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
Agradients/loss/absolute_difference/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/loss/absolute_difference/Mul_grad/Shape3gradients/loss/absolute_difference/Mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
/gradients/loss/absolute_difference/Mul_grad/MulMul0gradients/loss/absolute_difference/Sum_grad/Tile$loss/absolute_difference/ToFloat_3/x*
T0*'
_output_shapes
:���������
�
/gradients/loss/absolute_difference/Mul_grad/SumSum/gradients/loss/absolute_difference/Mul_grad/MulAgradients/loss/absolute_difference/Mul_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
3gradients/loss/absolute_difference/Mul_grad/ReshapeReshape/gradients/loss/absolute_difference/Mul_grad/Sum1gradients/loss/absolute_difference/Mul_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
1gradients/loss/absolute_difference/Mul_grad/Mul_1Mulloss/absolute_difference/Abs0gradients/loss/absolute_difference/Sum_grad/Tile*
T0*'
_output_shapes
:���������
�
1gradients/loss/absolute_difference/Mul_grad/Sum_1Sum1gradients/loss/absolute_difference/Mul_grad/Mul_1Cgradients/loss/absolute_difference/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
5gradients/loss/absolute_difference/Mul_grad/Reshape_1Reshape1gradients/loss/absolute_difference/Mul_grad/Sum_13gradients/loss/absolute_difference/Mul_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
�
<gradients/loss/absolute_difference/Mul_grad/tuple/group_depsNoOp4^gradients/loss/absolute_difference/Mul_grad/Reshape6^gradients/loss/absolute_difference/Mul_grad/Reshape_1
�
Dgradients/loss/absolute_difference/Mul_grad/tuple/control_dependencyIdentity3gradients/loss/absolute_difference/Mul_grad/Reshape=^gradients/loss/absolute_difference/Mul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/loss/absolute_difference/Mul_grad/Reshape*
T0*'
_output_shapes
:���������
�
Fgradients/loss/absolute_difference/Mul_grad/tuple/control_dependency_1Identity5gradients/loss/absolute_difference/Mul_grad/Reshape_1=^gradients/loss/absolute_difference/Mul_grad/tuple/group_deps*H
_class>
<:loc:@gradients/loss/absolute_difference/Mul_grad/Reshape_1*
T0*
_output_shapes
: 
�
Agradients/loss/absolute_difference/num_present_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
;gradients/loss/absolute_difference/num_present_grad/ReshapeReshapeIgradients/loss/absolute_difference/Select_grad/tuple/control_dependency_1Agradients/loss/absolute_difference/num_present_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes

:
�
9gradients/loss/absolute_difference/num_present_grad/ShapeShape6loss/absolute_difference/num_present/broadcast_weights*
out_type0*
T0*
_output_shapes
:
�
8gradients/loss/absolute_difference/num_present_grad/TileTile;gradients/loss/absolute_difference/num_present_grad/Reshape9gradients/loss/absolute_difference/num_present_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
g
*gradients/loss/add_5_grad/tuple/group_depsNoOp3^gradients/loss/add_6_grad/tuple/control_dependency
�
2gradients/loss/add_5_grad/tuple/control_dependencyIdentity2gradients/loss/add_6_grad/tuple/control_dependency+^gradients/loss/add_5_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_5_grad/tuple/control_dependency_1Identity2gradients/loss/add_6_grad/tuple/control_dependency+^gradients/loss/add_5_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_6_grad/MulMul4gradients/loss/add_6_grad/tuple/control_dependency_1loss/L2Loss_6*
T0*
_output_shapes
: 
�
gradients/loss/mul_6_grad/Mul_1Mul4gradients/loss/add_6_grad/tuple/control_dependency_1loss/mul_6/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_6_grad/tuple/group_depsNoOp^gradients/loss/mul_6_grad/Mul ^gradients/loss/mul_6_grad/Mul_1
�
2gradients/loss/mul_6_grad/tuple/control_dependencyIdentitygradients/loss/mul_6_grad/Mul+^gradients/loss/mul_6_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_6_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_6_grad/tuple/control_dependency_1Identitygradients/loss/mul_6_grad/Mul_1+^gradients/loss/mul_6_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_6_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_7_grad/mulMulVariable_7/read4gradients/loss/mul_7_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
�
Kgradients/loss/absolute_difference/num_present/broadcast_weights_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
�
Mgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Shape_1Shape@loss/absolute_difference/num_present/broadcast_weights/ones_like*
out_type0*
T0*
_output_shapes
:
�
[gradients/loss/absolute_difference/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsKgradients/loss/absolute_difference/num_present/broadcast_weights_grad/ShapeMgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Igradients/loss/absolute_difference/num_present/broadcast_weights_grad/MulMul8gradients/loss/absolute_difference/num_present_grad/Tile@loss/absolute_difference/num_present/broadcast_weights/ones_like*
T0*'
_output_shapes
:���������
�
Igradients/loss/absolute_difference/num_present/broadcast_weights_grad/SumSumIgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Mul[gradients/loss/absolute_difference/num_present/broadcast_weights_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Mgradients/loss/absolute_difference/num_present/broadcast_weights_grad/ReshapeReshapeIgradients/loss/absolute_difference/num_present/broadcast_weights_grad/SumKgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
�
Kgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Mul_1Mul+loss/absolute_difference/num_present/Select8gradients/loss/absolute_difference/num_present_grad/Tile*
T0*'
_output_shapes
:���������
�
Kgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Sum_1SumKgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Mul_1]gradients/loss/absolute_difference/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Ogradients/loss/absolute_difference/num_present/broadcast_weights_grad/Reshape_1ReshapeKgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Sum_1Mgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:���������
�
Vgradients/loss/absolute_difference/num_present/broadcast_weights_grad/tuple/group_depsNoOpN^gradients/loss/absolute_difference/num_present/broadcast_weights_grad/ReshapeP^gradients/loss/absolute_difference/num_present/broadcast_weights_grad/Reshape_1
�
^gradients/loss/absolute_difference/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityMgradients/loss/absolute_difference/num_present/broadcast_weights_grad/ReshapeW^gradients/loss/absolute_difference/num_present/broadcast_weights_grad/tuple/group_deps*`
_classV
TRloc:@gradients/loss/absolute_difference/num_present/broadcast_weights_grad/Reshape*
T0*
_output_shapes
: 
�
`gradients/loss/absolute_difference/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityOgradients/loss/absolute_difference/num_present/broadcast_weights_grad/Reshape_1W^gradients/loss/absolute_difference/num_present/broadcast_weights_grad/tuple/group_deps*b
_classX
VTloc:@gradients/loss/absolute_difference/num_present/broadcast_weights_grad/Reshape_1*
T0*'
_output_shapes
:���������
g
*gradients/loss/add_4_grad/tuple/group_depsNoOp3^gradients/loss/add_5_grad/tuple/control_dependency
�
2gradients/loss/add_4_grad/tuple/control_dependencyIdentity2gradients/loss/add_5_grad/tuple/control_dependency+^gradients/loss/add_4_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_4_grad/tuple/control_dependency_1Identity2gradients/loss/add_5_grad/tuple/control_dependency+^gradients/loss/add_4_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_5_grad/MulMul4gradients/loss/add_5_grad/tuple/control_dependency_1loss/L2Loss_5*
T0*
_output_shapes
: 
�
gradients/loss/mul_5_grad/Mul_1Mul4gradients/loss/add_5_grad/tuple/control_dependency_1loss/mul_5/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_5_grad/tuple/group_depsNoOp^gradients/loss/mul_5_grad/Mul ^gradients/loss/mul_5_grad/Mul_1
�
2gradients/loss/mul_5_grad/tuple/control_dependencyIdentitygradients/loss/mul_5_grad/Mul+^gradients/loss/mul_5_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_5_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_5_grad/tuple/control_dependency_1Identitygradients/loss/mul_5_grad/Mul_1+^gradients/loss/mul_5_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_5_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_6_grad/mulMulVariable_6/read4gradients/loss/mul_6_grad/tuple/control_dependency_1*
T0*
_output_shapes

:2
�
Ugradients/loss/absolute_difference/num_present/broadcast_weights/ones_like_grad/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
Sgradients/loss/absolute_difference/num_present/broadcast_weights/ones_like_grad/SumSum`gradients/loss/absolute_difference/num_present/broadcast_weights_grad/tuple/control_dependency_1Ugradients/loss/absolute_difference/num_present/broadcast_weights/ones_like_grad/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
g
*gradients/loss/add_3_grad/tuple/group_depsNoOp3^gradients/loss/add_4_grad/tuple/control_dependency
�
2gradients/loss/add_3_grad/tuple/control_dependencyIdentity2gradients/loss/add_4_grad/tuple/control_dependency+^gradients/loss/add_3_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_3_grad/tuple/control_dependency_1Identity2gradients/loss/add_4_grad/tuple/control_dependency+^gradients/loss/add_3_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_4_grad/MulMul4gradients/loss/add_4_grad/tuple/control_dependency_1loss/L2Loss_4*
T0*
_output_shapes
: 
�
gradients/loss/mul_4_grad/Mul_1Mul4gradients/loss/add_4_grad/tuple/control_dependency_1loss/mul_4/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_4_grad/tuple/group_depsNoOp^gradients/loss/mul_4_grad/Mul ^gradients/loss/mul_4_grad/Mul_1
�
2gradients/loss/mul_4_grad/tuple/control_dependencyIdentitygradients/loss/mul_4_grad/Mul+^gradients/loss/mul_4_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_4_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_4_grad/tuple/control_dependency_1Identitygradients/loss/mul_4_grad/Mul_1+^gradients/loss/mul_4_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_4_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_5_grad/mulMulVariable_5/read4gradients/loss/mul_5_grad/tuple/control_dependency_1*
T0*
_output_shapes
:2
g
*gradients/loss/add_2_grad/tuple/group_depsNoOp3^gradients/loss/add_3_grad/tuple/control_dependency
�
2gradients/loss/add_2_grad/tuple/control_dependencyIdentity2gradients/loss/add_3_grad/tuple/control_dependency+^gradients/loss/add_2_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_2_grad/tuple/control_dependency_1Identity2gradients/loss/add_3_grad/tuple/control_dependency+^gradients/loss/add_2_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_3_grad/MulMul4gradients/loss/add_3_grad/tuple/control_dependency_1loss/L2Loss_3*
T0*
_output_shapes
: 
�
gradients/loss/mul_3_grad/Mul_1Mul4gradients/loss/add_3_grad/tuple/control_dependency_1loss/mul_3/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_3_grad/tuple/group_depsNoOp^gradients/loss/mul_3_grad/Mul ^gradients/loss/mul_3_grad/Mul_1
�
2gradients/loss/mul_3_grad/tuple/control_dependencyIdentitygradients/loss/mul_3_grad/Mul+^gradients/loss/mul_3_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_3_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_3_grad/tuple/control_dependency_1Identitygradients/loss/mul_3_grad/Mul_1+^gradients/loss/mul_3_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_3_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_4_grad/mulMulVariable_4/read4gradients/loss/mul_4_grad/tuple/control_dependency_1*
T0*
_output_shapes

:d2
�
0gradients/loss/absolute_difference/Abs_grad/SignSignloss/absolute_difference/Sub*
T0*'
_output_shapes
:���������
�
/gradients/loss/absolute_difference/Abs_grad/mulMulDgradients/loss/absolute_difference/Mul_grad/tuple/control_dependency0gradients/loss/absolute_difference/Abs_grad/Sign*
T0*'
_output_shapes
:���������
g
*gradients/loss/add_1_grad/tuple/group_depsNoOp3^gradients/loss/add_2_grad/tuple/control_dependency
�
2gradients/loss/add_1_grad/tuple/control_dependencyIdentity2gradients/loss/add_2_grad/tuple/control_dependency+^gradients/loss/add_1_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
4gradients/loss/add_1_grad/tuple/control_dependency_1Identity2gradients/loss/add_2_grad/tuple/control_dependency+^gradients/loss/add_1_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_2_grad/MulMul4gradients/loss/add_2_grad/tuple/control_dependency_1loss/L2Loss_2*
T0*
_output_shapes
: 
�
gradients/loss/mul_2_grad/Mul_1Mul4gradients/loss/add_2_grad/tuple/control_dependency_1loss/mul_2/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_2_grad/tuple/group_depsNoOp^gradients/loss/mul_2_grad/Mul ^gradients/loss/mul_2_grad/Mul_1
�
2gradients/loss/mul_2_grad/tuple/control_dependencyIdentitygradients/loss/mul_2_grad/Mul+^gradients/loss/mul_2_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_2_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_2_grad/tuple/control_dependency_1Identitygradients/loss/mul_2_grad/Mul_1+^gradients/loss/mul_2_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_2_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_3_grad/mulMulVariable_3/read4gradients/loss/mul_3_grad/tuple/control_dependency_1*
T0*
_output_shapes
:d
x
1gradients/loss/absolute_difference/Sub_grad/ShapeShapeone_hot*
out_type0*
T0*
_output_shapes
:
y
3gradients/loss/absolute_difference/Sub_grad/Shape_1Shapelabels*
out_type0*
T0*
_output_shapes
:
�
Agradients/loss/absolute_difference/Sub_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/loss/absolute_difference/Sub_grad/Shape3gradients/loss/absolute_difference/Sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
/gradients/loss/absolute_difference/Sub_grad/SumSum/gradients/loss/absolute_difference/Abs_grad/mulAgradients/loss/absolute_difference/Sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
3gradients/loss/absolute_difference/Sub_grad/ReshapeReshape/gradients/loss/absolute_difference/Sub_grad/Sum1gradients/loss/absolute_difference/Sub_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
1gradients/loss/absolute_difference/Sub_grad/Sum_1Sum/gradients/loss/absolute_difference/Abs_grad/mulCgradients/loss/absolute_difference/Sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
/gradients/loss/absolute_difference/Sub_grad/NegNeg1gradients/loss/absolute_difference/Sub_grad/Sum_1*
T0*
_output_shapes
:
�
5gradients/loss/absolute_difference/Sub_grad/Reshape_1Reshape/gradients/loss/absolute_difference/Sub_grad/Neg3gradients/loss/absolute_difference/Sub_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:���������
�
<gradients/loss/absolute_difference/Sub_grad/tuple/group_depsNoOp4^gradients/loss/absolute_difference/Sub_grad/Reshape6^gradients/loss/absolute_difference/Sub_grad/Reshape_1
�
Dgradients/loss/absolute_difference/Sub_grad/tuple/control_dependencyIdentity3gradients/loss/absolute_difference/Sub_grad/Reshape=^gradients/loss/absolute_difference/Sub_grad/tuple/group_deps*F
_class<
:8loc:@gradients/loss/absolute_difference/Sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
Fgradients/loss/absolute_difference/Sub_grad/tuple/control_dependency_1Identity5gradients/loss/absolute_difference/Sub_grad/Reshape_1=^gradients/loss/absolute_difference/Sub_grad/tuple/group_deps*H
_class>
<:loc:@gradients/loss/absolute_difference/Sub_grad/Reshape_1*
T0*'
_output_shapes
:���������
e
(gradients/loss/add_grad/tuple/group_depsNoOp3^gradients/loss/add_1_grad/tuple/control_dependency
�
0gradients/loss/add_grad/tuple/control_dependencyIdentity2gradients/loss/add_1_grad/tuple/control_dependency)^gradients/loss/add_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
2gradients/loss/add_grad/tuple/control_dependency_1Identity2gradients/loss/add_1_grad/tuple/control_dependency)^gradients/loss/add_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
T0*
_output_shapes
: 
�
gradients/loss/mul_1_grad/MulMul4gradients/loss/add_1_grad/tuple/control_dependency_1loss/L2Loss_1*
T0*
_output_shapes
: 
�
gradients/loss/mul_1_grad/Mul_1Mul4gradients/loss/add_1_grad/tuple/control_dependency_1loss/mul_1/x*
T0*
_output_shapes
: 
t
*gradients/loss/mul_1_grad/tuple/group_depsNoOp^gradients/loss/mul_1_grad/Mul ^gradients/loss/mul_1_grad/Mul_1
�
2gradients/loss/mul_1_grad/tuple/control_dependencyIdentitygradients/loss/mul_1_grad/Mul+^gradients/loss/mul_1_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_1_grad/Mul*
T0*
_output_shapes
: 
�
4gradients/loss/mul_1_grad/tuple/control_dependency_1Identitygradients/loss/mul_1_grad/Mul_1+^gradients/loss/mul_1_grad/tuple/group_deps*2
_class(
&$loc:@gradients/loss/mul_1_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_2_grad/mulMulVariable_2/read4gradients/loss/mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	�d
�
gradients/loss/mul_grad/MulMul2gradients/loss/add_grad/tuple/control_dependency_1loss/L2Loss*
T0*
_output_shapes
: 
�
gradients/loss/mul_grad/Mul_1Mul2gradients/loss/add_grad/tuple/control_dependency_1
loss/mul/x*
T0*
_output_shapes
: 
n
(gradients/loss/mul_grad/tuple/group_depsNoOp^gradients/loss/mul_grad/Mul^gradients/loss/mul_grad/Mul_1
�
0gradients/loss/mul_grad/tuple/control_dependencyIdentitygradients/loss/mul_grad/Mul)^gradients/loss/mul_grad/tuple/group_deps*.
_class$
" loc:@gradients/loss/mul_grad/Mul*
T0*
_output_shapes
: 
�
2gradients/loss/mul_grad/tuple/control_dependency_1Identitygradients/loss/mul_grad/Mul_1)^gradients/loss/mul_grad/tuple/group_deps*0
_class&
$"loc:@gradients/loss/mul_grad/Mul_1*
T0*
_output_shapes
: 
�
 gradients/loss/L2Loss_1_grad/mulMulVariable_1/read4gradients/loss/mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
gradients/loss/L2Loss_grad/mulMulVariable/read2gradients/loss/mul_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	�
�
"Variable/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable*
valueB	�*���=*
_output_shapes
:	�
�
Variable/Adagrad
VariableV2*
	container *
_output_shapes
:	�*
dtype0*
shape:	�*
_class
loc:@Variable*
shared_name 
�
Variable/Adagrad/AssignAssignVariable/Adagrad"Variable/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
:	�
z
Variable/Adagrad/readIdentityVariable/Adagrad*
_class
loc:@Variable*
T0*
_output_shapes
:	�
�
$Variable_1/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_1*
valueB�*���=*
_output_shapes	
:�
�
Variable_1/Adagrad
VariableV2*
	container *
_output_shapes	
:�*
dtype0*
shape:�*
_class
loc:@Variable_1*
shared_name 
�
Variable_1/Adagrad/AssignAssignVariable_1/Adagrad$Variable_1/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes	
:�
|
Variable_1/Adagrad/readIdentityVariable_1/Adagrad*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
�
$Variable_2/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_2*
valueB	�d*���=*
_output_shapes
:	�d
�
Variable_2/Adagrad
VariableV2*
	container *
_output_shapes
:	�d*
dtype0*
shape:	�d*
_class
loc:@Variable_2*
shared_name 
�
Variable_2/Adagrad/AssignAssignVariable_2/Adagrad$Variable_2/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
:	�d
�
Variable_2/Adagrad/readIdentityVariable_2/Adagrad*
_class
loc:@Variable_2*
T0*
_output_shapes
:	�d
�
$Variable_3/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_3*
valueBd*���=*
_output_shapes
:d
�
Variable_3/Adagrad
VariableV2*
	container *
_output_shapes
:d*
dtype0*
shape:d*
_class
loc:@Variable_3*
shared_name 
�
Variable_3/Adagrad/AssignAssignVariable_3/Adagrad$Variable_3/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:d
{
Variable_3/Adagrad/readIdentityVariable_3/Adagrad*
_class
loc:@Variable_3*
T0*
_output_shapes
:d
�
$Variable_4/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_4*
valueBd2*���=*
_output_shapes

:d2
�
Variable_4/Adagrad
VariableV2*
	container *
_output_shapes

:d2*
dtype0*
shape
:d2*
_class
loc:@Variable_4*
shared_name 
�
Variable_4/Adagrad/AssignAssignVariable_4/Adagrad$Variable_4/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:d2

Variable_4/Adagrad/readIdentityVariable_4/Adagrad*
_class
loc:@Variable_4*
T0*
_output_shapes

:d2
�
$Variable_5/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_5*
valueB2*���=*
_output_shapes
:2
�
Variable_5/Adagrad
VariableV2*
	container *
_output_shapes
:2*
dtype0*
shape:2*
_class
loc:@Variable_5*
shared_name 
�
Variable_5/Adagrad/AssignAssignVariable_5/Adagrad$Variable_5/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes
:2
{
Variable_5/Adagrad/readIdentityVariable_5/Adagrad*
_class
loc:@Variable_5*
T0*
_output_shapes
:2
�
$Variable_6/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_6*
valueB2*���=*
_output_shapes

:2
�
Variable_6/Adagrad
VariableV2*
	container *
_output_shapes

:2*
dtype0*
shape
:2*
_class
loc:@Variable_6*
shared_name 
�
Variable_6/Adagrad/AssignAssignVariable_6/Adagrad$Variable_6/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes

:2

Variable_6/Adagrad/readIdentityVariable_6/Adagrad*
_class
loc:@Variable_6*
T0*
_output_shapes

:2
�
$Variable_7/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_7*
valueB*���=*
_output_shapes
:
�
Variable_7/Adagrad
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@Variable_7*
shared_name 
�
Variable_7/Adagrad/AssignAssignVariable_7/Adagrad$Variable_7/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:
{
Variable_7/Adagrad/readIdentityVariable_7/Adagrad*
_class
loc:@Variable_7*
T0*
_output_shapes
:
�
$Variable_8/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_8*
valueB*���=*
_output_shapes

:
�
Variable_8/Adagrad
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*
_class
loc:@Variable_8*
shared_name 
�
Variable_8/Adagrad/AssignAssignVariable_8/Adagrad$Variable_8/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_8*
use_locking(*
T0*
_output_shapes

:

Variable_8/Adagrad/readIdentityVariable_8/Adagrad*
_class
loc:@Variable_8*
T0*
_output_shapes

:
�
$Variable_9/Adagrad/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_9*
valueB*���=*
_output_shapes
:
�
Variable_9/Adagrad
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*
_class
loc:@Variable_9*
shared_name 
�
Variable_9/Adagrad/AssignAssignVariable_9/Adagrad$Variable_9/Adagrad/Initializer/Const*
validate_shape(*
_class
loc:@Variable_9*
use_locking(*
T0*
_output_shapes
:
{
Variable_9/Adagrad/readIdentityVariable_9/Adagrad*
_class
loc:@Variable_9*
T0*
_output_shapes
:
Z
Adagrad/learning_rateConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
�
$Adagrad/update_Variable/ApplyAdagradApplyAdagradVariableVariable/AdagradAdagrad/learning_rategradients/loss/L2Loss_grad/mul*
_class
loc:@Variable*
use_locking( *
T0*
_output_shapes
:	�
�
&Adagrad/update_Variable_1/ApplyAdagradApplyAdagrad
Variable_1Variable_1/AdagradAdagrad/learning_rate gradients/loss/L2Loss_1_grad/mul*
_class
loc:@Variable_1*
use_locking( *
T0*
_output_shapes	
:�
�
&Adagrad/update_Variable_2/ApplyAdagradApplyAdagrad
Variable_2Variable_2/AdagradAdagrad/learning_rate gradients/loss/L2Loss_2_grad/mul*
_class
loc:@Variable_2*
use_locking( *
T0*
_output_shapes
:	�d
�
&Adagrad/update_Variable_3/ApplyAdagradApplyAdagrad
Variable_3Variable_3/AdagradAdagrad/learning_rate gradients/loss/L2Loss_3_grad/mul*
_class
loc:@Variable_3*
use_locking( *
T0*
_output_shapes
:d
�
&Adagrad/update_Variable_4/ApplyAdagradApplyAdagrad
Variable_4Variable_4/AdagradAdagrad/learning_rate gradients/loss/L2Loss_4_grad/mul*
_class
loc:@Variable_4*
use_locking( *
T0*
_output_shapes

:d2
�
&Adagrad/update_Variable_5/ApplyAdagradApplyAdagrad
Variable_5Variable_5/AdagradAdagrad/learning_rate gradients/loss/L2Loss_5_grad/mul*
_class
loc:@Variable_5*
use_locking( *
T0*
_output_shapes
:2
�
&Adagrad/update_Variable_6/ApplyAdagradApplyAdagrad
Variable_6Variable_6/AdagradAdagrad/learning_rate gradients/loss/L2Loss_6_grad/mul*
_class
loc:@Variable_6*
use_locking( *
T0*
_output_shapes

:2
�
&Adagrad/update_Variable_7/ApplyAdagradApplyAdagrad
Variable_7Variable_7/AdagradAdagrad/learning_rate gradients/loss/L2Loss_7_grad/mul*
_class
loc:@Variable_7*
use_locking( *
T0*
_output_shapes
:
�
&Adagrad/update_Variable_8/ApplyAdagradApplyAdagrad
Variable_8Variable_8/AdagradAdagrad/learning_rate gradients/loss/L2Loss_8_grad/mul*
_class
loc:@Variable_8*
use_locking( *
T0*
_output_shapes

:
�
&Adagrad/update_Variable_9/ApplyAdagradApplyAdagrad
Variable_9Variable_9/AdagradAdagrad/learning_rate gradients/loss/L2Loss_9_grad/mul*
_class
loc:@Variable_9*
use_locking( *
T0*
_output_shapes
:
�
Adagrad/updateNoOp%^Adagrad/update_Variable/ApplyAdagrad'^Adagrad/update_Variable_1/ApplyAdagrad'^Adagrad/update_Variable_2/ApplyAdagrad'^Adagrad/update_Variable_3/ApplyAdagrad'^Adagrad/update_Variable_4/ApplyAdagrad'^Adagrad/update_Variable_5/ApplyAdagrad'^Adagrad/update_Variable_6/ApplyAdagrad'^Adagrad/update_Variable_7/ApplyAdagrad'^Adagrad/update_Variable_8/ApplyAdagrad'^Adagrad/update_Variable_9/ApplyAdagrad
�
Adagrad/valueConst^Adagrad/update*
dtype0*
_class
loc:@Variable_10*
value	B :*
_output_shapes
: 
�
Adagrad	AssignAddVariable_10Adagrad/value*
_class
loc:@Variable_10*
use_locking( *
T0*
_output_shapes
: 
v
Merge/MergeSummaryMergeSummaryprediction_lossreg_loss
total_loss
accuracy_1*
N*
_output_shapes
: ""
train_op
	
Adagrad"�
	variables��
B

Variable:0Variable/AssignVariable/read:02truncated_normal:0
=
Variable_1:0Variable_1/AssignVariable_1/read:02zeros:0
J
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:0
?
Variable_3:0Variable_3/AssignVariable_3/read:02	zeros_1:0
J
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_2:0
?
Variable_5:0Variable_5/AssignVariable_5/read:02	zeros_2:0
J
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_3:0
?
Variable_7:0Variable_7/AssignVariable_7/read:02	zeros_3:0
J
Variable_8:0Variable_8/AssignVariable_8/read:02truncated_normal_4:0
?
Variable_9:0Variable_9/AssignVariable_9/read:02	zeros_4:0
T
Variable_10:0Variable_10/AssignVariable_10/read:02Variable_10/initial_value:0
l
Variable/Adagrad:0Variable/Adagrad/AssignVariable/Adagrad/read:02$Variable/Adagrad/Initializer/Const:0
t
Variable_1/Adagrad:0Variable_1/Adagrad/AssignVariable_1/Adagrad/read:02&Variable_1/Adagrad/Initializer/Const:0
t
Variable_2/Adagrad:0Variable_2/Adagrad/AssignVariable_2/Adagrad/read:02&Variable_2/Adagrad/Initializer/Const:0
t
Variable_3/Adagrad:0Variable_3/Adagrad/AssignVariable_3/Adagrad/read:02&Variable_3/Adagrad/Initializer/Const:0
t
Variable_4/Adagrad:0Variable_4/Adagrad/AssignVariable_4/Adagrad/read:02&Variable_4/Adagrad/Initializer/Const:0
t
Variable_5/Adagrad:0Variable_5/Adagrad/AssignVariable_5/Adagrad/read:02&Variable_5/Adagrad/Initializer/Const:0
t
Variable_6/Adagrad:0Variable_6/Adagrad/AssignVariable_6/Adagrad/read:02&Variable_6/Adagrad/Initializer/Const:0
t
Variable_7/Adagrad:0Variable_7/Adagrad/AssignVariable_7/Adagrad/read:02&Variable_7/Adagrad/Initializer/Const:0
t
Variable_8/Adagrad:0Variable_8/Adagrad/AssignVariable_8/Adagrad/read:02&Variable_8/Adagrad/Initializer/Const:0
t
Variable_9/Adagrad:0Variable_9/Adagrad/AssignVariable_9/Adagrad/read:02&Variable_9/Adagrad/Initializer/Const:0".
losses$
"
 loss/absolute_difference/value:0"J
	summaries=
;
prediction_loss:0

reg_loss:0
total_loss:0
accuracy_1:0"�
trainable_variables��
B

Variable:0Variable/AssignVariable/read:02truncated_normal:0
=
Variable_1:0Variable_1/AssignVariable_1/read:02zeros:0
J
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal_1:0
?
Variable_3:0Variable_3/AssignVariable_3/read:02	zeros_1:0
J
Variable_4:0Variable_4/AssignVariable_4/read:02truncated_normal_2:0
?
Variable_5:0Variable_5/AssignVariable_5/read:02	zeros_2:0
J
Variable_6:0Variable_6/AssignVariable_6/read:02truncated_normal_3:0
?
Variable_7:0Variable_7/AssignVariable_7/read:02	zeros_3:0
J
Variable_8:0Variable_8/AssignVariable_8/read:02truncated_normal_4:0
?
Variable_9:0Variable_9/AssignVariable_9/read:02	zeros_4:0
T
Variable_10:0Variable_10/AssignVariable_10/read:02Variable_10/initial_value:0B  :Z       o��		��)_��A*O

prediction_loss   ?

reg_loss�x<


total_loss0�?


accuracy_1   ?����\       ����	Ӡ�)_��A*O

prediction_loss)\?

reg_loss�x<


total_lossX<?


accuracy_1�G�>"\       ����	���)_��A*O

prediction_loss���>

reg_loss�x<


total_loss��>


accuracy_1�?Z�vu\       ����	���)_��A*O

prediction_loss���>

reg_loss�x<


total_loss��>


accuracy_1�?5���\       ����	��)_��A*O

prediction_loss��>

reg_loss�x<


total_loss|E�>


accuracy_1q=
?�}�\       ����	�4�)_��A*O

prediction_loss��>

reg_loss�x<


total_loss|E�>


accuracy_1q=
?O�)\       ����	�c�)_��A*O

prediction_loss
ף>

reg_loss�x<


total_lossg��>


accuracy_1{.?�XH\       ����	M��)_��A*O

prediction_loss   ?

reg_loss�x<


total_loss.�?


accuracy_1   ?w�I\       ����	D��)_��A*O

prediction_loss���>

reg_loss�x<


total_loss��>


accuracy_1�?.F.�\       ����	;��)_��A	*O

prediction_loss=
�>

reg_loss�x<


total_loss���>


accuracy_1�z?��jY\       ����	D��)_��A
*O

prediction_loss���>

reg_lossux<


total_loss��>


accuracy_1�?DA�\       ����	��)_��A*O

prediction_loss�G�>

reg_losskx<


total_loss	�>


accuracy_1)\?u2\       ����	0/�)_��A*O

prediction_loss��>

reg_loss`x<


total_losszE�>


accuracy_1q=
?�?�k\       ����	�O�)_��A*O

prediction_loss�G�>

reg_lossUx<


total_loss	�>


accuracy_1)\?�\       ����	-y�)_��A*O

prediction_loss   ?

reg_lossJx<


total_loss-�?


accuracy_1   ?=4.0\       ����	���)_��A*O

prediction_lossq=
?

reg_loss@x<


total_loss�?


accuracy_1��>��%\       ����	,.�)_��A*O

prediction_loss)\?

reg_loss5x<


total_lossV<?


accuracy_1�G�>��Ζ\       ����	Q�)_��A*O

prediction_loss{.?

reg_loss+x<


total_loss��1?


accuracy_1
ף>�Q�]\       ����	�r�)_��A*O

prediction_loss=
�>

reg_loss!x<


total_loss���>


accuracy_1�z?��o\       ����	d��)_��A*O

prediction_loss��?

reg_lossx<


total_loss�y?


accuracy_1���> ��\       ����	F��)_��A*O

prediction_loss=
�>

reg_lossx<


total_loss���>


accuracy_1�z?NB�\       ����	d��)_��A*O

prediction_loss�z?

reg_lossx<


total_loss[?


accuracy_1=
�>�Q�C\       ����	���)_��A*O

prediction_loss���>

reg_loss�
x<


total_loss��>


accuracy_1�?9,�\       ����	(�)_��A*O

prediction_loss   ?

reg_loss�
x<


total_loss,�?


accuracy_1   ?�'�\       ����	�,�)_��A*O

prediction_loss��?

reg_loss�
x<


total_loss�y?


accuracy_1���>mۯ\       ����	oJ�)_��A*O

prediction_loss�G�>

reg_loss�
x<


total_loss�>


accuracy_1)\?X���\       ����	�g�)_��A*O

prediction_loss�?

reg_loss�
x<


total_loss��?


accuracy_1���>�]>\       ����	���)_��A*O

prediction_loss�?

reg_loss�
x<


total_loss��?


accuracy_1���>}0թ\       ����	%��)_��A*O

prediction_loss���>

reg_loss�
x<


total_loss#��>


accuracy_1��?Z糒\       ����	W��)_��A*O

prediction_loss��>

reg_loss�
x<


total_losstE�>


accuracy_1q=
?1��*\       ����	��)_��A*O

prediction_lossq=
?

reg_loss�
x<


total_loss�?


accuracy_1��>~�8\       ����	��)_��A*O

prediction_lossR�?

reg_loss�
x<


total_loss|�"?


accuracy_1\��>Oo]y\       ����	37�)_��A *O

prediction_loss��?

reg_loss�
x<


total_loss�y?


accuracy_1���>FA�(\       ����	�T�)_��A!*O

prediction_lossq=
?

reg_loss�
x<


total_loss�?


accuracy_1��>(ޅ(\       ����	o�)_��A"*O

prediction_lossR�?

reg_loss|
x<


total_loss|�"?


accuracy_1\��>/�t!\       ����	'��)_��A#*O

prediction_loss   ?

reg_lossr
x<


total_loss*�?


accuracy_1   ?���u\       ����	l��)_��A$*O

prediction_loss\��>

reg_lossi
x<


total_loss�O�>


accuracy_1R�?��\       ����	��)_��A%*O

prediction_lossR�?

reg_loss^
x<


total_loss{�"?


accuracy_1\��>�L�I\       ����	3�)_��A&*O

prediction_loss���>

reg_lossT
x<


total_loss��>


accuracy_1�?"���\       ����	78�)_��A'*O

prediction_loss)\?

reg_lossI
x<


total_lossR<?


accuracy_1�G�>�?t\       ����		S�)_��A(*O

prediction_loss�z?

reg_lossA
x<


total_loss
[?


accuracy_1=
�>�i�\       ����	*o�)_��A)*O

prediction_lossq=
?

reg_loss6
x<


total_loss�?


accuracy_1��>g���\       ����	z��)_��A**O

prediction_loss��>

reg_loss+
x<


total_losspE�>


accuracy_1q=
?8#�\       ����	Ƨ�)_��A+*O

prediction_loss���>

reg_loss!
x<


total_loss��>


accuracy_1��?�\X�\       ����	D��)_��A,*O

prediction_loss�?

reg_loss
x<


total_loss��?


accuracy_1���>0�D\       ����	���)_��A-*O

prediction_loss���>

reg_loss
x<


total_loss߂�>


accuracy_1�?���\       ����	���)_��A.*O

prediction_loss�?

reg_loss
x<


total_loss��?


accuracy_1���>�K#�\       ����	�)_��A/*O

prediction_lossR�?

reg_loss�	x<


total_lossz�"?


accuracy_1\��>��x\       ����	�(�)_��A0*O

prediction_loss)\?

reg_loss�	x<


total_lossQ<?


accuracy_1�G�>���\       ����	�B�)_��A1*O

prediction_loss�?

reg_loss�	x<


total_loss��?


accuracy_1���>A�j�\       ����	1[�)_��A2*O

prediction_loss���>

reg_loss�	x<


total_lossނ�>


accuracy_1�?�`�\       ����	)y�)_��A3*O

prediction_loss��>

reg_loss�	x<


total_lossmE�>


accuracy_1q=
?Y�
`\       ����	���)_��A4*O

prediction_loss��>

reg_loss�	x<


total_lossmE�>


accuracy_1q=
?��:\       ����	���)_��A5*O

prediction_loss�?

reg_loss�	x<


total_loss��?


accuracy_1���>=f�\       ����	���)_��A6*O

prediction_loss�?

reg_loss�	x<


total_loss��?


accuracy_1���> L��\       ����	��)_��A7*O

prediction_loss�?

reg_loss�	x<


total_loss��?


accuracy_1���>�6��\       ����	��)_��A8*O

prediction_loss)\?

reg_loss�	x<


total_lossO<?


accuracy_1�G�>�&W�\       ����	#�)_��A9*O

prediction_loss�z?

reg_loss�	x<


total_loss[?


accuracy_1=
�>��L\       ����	o+�)_��A:*O

prediction_loss)\?

reg_loss�	x<


total_lossO<?


accuracy_1�G�>z���\       ����	xA�)_��A;*O

prediction_lossq=
?

reg_loss~	x<


total_loss�?


accuracy_1��>�#|�\       ����	SZ�)_��A<*O

prediction_loss�?

reg_lossr	x<


total_loss��?


accuracy_1���>�qH\       ����	er�)_��A=*O

prediction_loss��>

reg_lossf	x<


total_lossjE�>


accuracy_1q=
?R*-U\       ����	a��)_��A>*O

prediction_lossq=
?

reg_loss]	x<


total_loss�?


accuracy_1��>|�O\       ����	��)_��A?*O

prediction_loss   ?

reg_lossT	x<


total_loss%�?


accuracy_1   ?0���\       ����	c��)_��A@*O

prediction_loss=
�>

reg_lossI	x<


total_loss���>


accuracy_1�z?s�IA\       ����	��)_��AA*O

prediction_loss�z?

reg_loss?	x<


total_loss[?


accuracy_1=
�>f(�\       ����	`��)_��AB*O

prediction_loss��>

reg_loss3	x<


total_lossiE�>


accuracy_1q=
?����\       ����	���)_��AC*O

prediction_loss�?

reg_loss*	x<


total_loss��?


accuracy_1���>*Ř\       ����	��)_��AD*O

prediction_loss�G�>

reg_loss 	x<


total_loss��>


accuracy_1)\?A��\       ����	1�)_��AE*O

prediction_loss��>

reg_loss	x<


total_losshE�>


accuracy_1q=
?چq\       ����	0L�)_��AF*O

prediction_loss
�#?

reg_loss	x<


total_loss.�'?


accuracy_1�Q�>|_YF\       ����	
d�)_��AG*O

prediction_loss   ?

reg_loss 	x<


total_loss$�?


accuracy_1   ?I��\       ����	F{�)_��AH*O

prediction_lossq=
?

reg_loss�x<


total_loss�?


accuracy_1��>�Q�\       ����	ɑ�)_��AI*O

prediction_loss���>

reg_loss�x<


total_loss��>


accuracy_1��?o,4;\       ����	���)_��AJ*O

prediction_loss�z?

reg_loss�x<


total_loss[?


accuracy_1=
�>��k\       ����	���)_��AK*O

prediction_loss��>

reg_loss�x<


total_loss�ތ>


accuracy_1�p=?��~�\       ����	��)_��AL*O

prediction_loss�Q8?

reg_loss�x<


total_loss2<?


accuracy_1)\�>�x��\       ����	���)_��AM*O

prediction_loss=
�>

reg_loss�x<


total_loss���>


accuracy_1�z?�1)\       ����	J
�)_��AN*O

prediction_loss���>

reg_loss�x<


total_lossՂ�>


accuracy_1�?E�n\       ����	#�)_��AO*O

prediction_loss   ?

reg_loss�x<


total_loss#�?


accuracy_1   ?4u \       ����	=�)_��AP*O

prediction_loss)\?

reg_loss�x<


total_lossL<?


accuracy_1�G�>����\       ����	�S�)_��AQ*O

prediction_loss   ?

reg_loss�x<


total_loss"�?


accuracy_1   ?_>�P\       ����	�m�)_��AR*O

prediction_loss�G�>

reg_loss�x<


total_loss��>


accuracy_1)\?.k%\       ����	��)_��AS*O

prediction_loss���>

reg_loss�x<


total_lossӂ�>


accuracy_1�?�²�\       ����	���)_��AT*O

prediction_loss)\?

reg_lossxx<


total_lossK<?


accuracy_1�G�>�_�]\       ����	��)_��AU*O

prediction_loss�z?

reg_lossnx<


total_loss[?


accuracy_1=
�>�e�\       ����	��)_��AV*O

prediction_loss��>

reg_lossdx<


total_lossbE�>


accuracy_1q=
?ŝq~\       ����	���)_��AW*O

prediction_loss�?

reg_loss\x<


total_loss��?


accuracy_1���>��e\       ����	���)_��AX*O

prediction_loss   ?

reg_lossOx<


total_loss!�?


accuracy_1   ?e0��\       ����	��)_��AY*O

prediction_loss=
�>

reg_lossEx<


total_loss��>


accuracy_1�z?LJ�\       ����	4�)_��AZ*O

prediction_loss�G�>

reg_loss:x<


total_loss��>


accuracy_1)\?��#\       ����	�K�)_��A[*O

prediction_loss���>

reg_loss1x<


total_loss��>


accuracy_1��?��mv\       ����	5b�)_��A\*O

prediction_loss�G�>

reg_loss%x<


total_loss��>


accuracy_1)\?�4��\       ����	Wx�)_��A]*O

prediction_loss��(?

reg_lossx<


total_loss��,?


accuracy_1{�>W	�#\       ����	��)_��A^*O

prediction_loss��>

reg_lossx<


total_loss`E�>


accuracy_1q=
?k7��\       ����	��)_��A_*O

prediction_lossq=
?

reg_lossx<


total_loss�?


accuracy_1��>�Ac\       ����	���)_��A`*O

prediction_loss���>

reg_loss�x<


total_lossς�>


accuracy_1�?�J��\       ����	���)_��Aa*O

prediction_loss)\?

reg_loss�x<


total_lossI<?


accuracy_1�G�>�!��\       ����	���)_��Ab*O

prediction_loss�?

reg_loss�x<


total_loss��?


accuracy_1���>�4\       ����	p	�)_��Ac*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?��[\       ����	�#�)_��Ad*O

prediction_loss���>

reg_loss�x<


total_loss΂�>


accuracy_1�?���\       ����	�o�)_��Ae*O

prediction_loss)\?

reg_loss�x<


total_lossH<?


accuracy_1�G�>XZ��\       ����	U��)_��Af*O

prediction_loss=
�>

reg_loss�x<


total_loss{��>


accuracy_1�z?��p�\       ����	���)_��Ag*O

prediction_lossq=
?

reg_loss�x<


total_loss�?


accuracy_1��>TI�\       ����	=��)_��Ah*O

prediction_lossR�?

reg_loss�x<


total_lossq�"?


accuracy_1\��>w�"r\       ����	1��)_��Ai*O

prediction_loss=
�>

reg_loss�x<


total_lossz��>


accuracy_1�z?#{D:\       ����	���)_��Aj*O

prediction_loss���>

reg_loss�x<


total_loss˂�>


accuracy_1�?��{u\       ����	�)_��Ak*O

prediction_loss���>

reg_loss�x<


total_loss˂�>


accuracy_1�?m��\       ����	W$�)_��Al*O

prediction_loss=
�>

reg_losswx<


total_lossy��>


accuracy_1�z?����\       ����	cC�)_��Am*O

prediction_loss   ?

reg_lossox<


total_loss�?


accuracy_1   ?дY�\       ����	�a�)_��An*O

prediction_loss��?

reg_losscx<


total_loss�y?


accuracy_1���>���\       ����	}�)_��Ao*O

prediction_loss�z?

reg_lossYx<


total_loss�Z?


accuracy_1=
�>���\       ����	/��)_��Ap*O

prediction_loss�?

reg_lossLx<


total_loss��?


accuracy_1���>��P\       ����	���)_��Aq*O

prediction_loss���>

reg_lossDx<


total_loss��>


accuracy_1��?ID \       ����	7��)_��Ar*O

prediction_loss   ?

reg_loss8x<


total_loss�?


accuracy_1   ?
�d�\       ����	<��)_��As*O

prediction_lossq=
?

reg_loss.x<


total_loss�?


accuracy_1��>;�&�\       ����	�)_��At*O

prediction_loss{.?

reg_loss#x<


total_loss��1?


accuracy_1
ף>\�ľ\       ����	�#�)_��Au*O

prediction_loss��>

reg_lossx<


total_lossXE�>


accuracy_1q=
?���\       ����	=�)_��Av*O

prediction_loss   ?

reg_lossx<


total_loss�?


accuracy_1   ?�A`�\       ����	�W�)_��Aw*O

prediction_loss���>

reg_lossx<


total_lossǂ�>


accuracy_1�?��ʘ\       ����	ǀ�)_��Ax*O

prediction_loss��>

reg_loss�x<


total_lossWE�>


accuracy_1q=
?|E�\       ����	ܡ�)_��Ay*O

prediction_lossR�?

reg_loss�x<


total_lossn�"?


accuracy_1\��>Y�?\       ����	n��)_��Az*O

prediction_loss�G�>

reg_loss�x<


total_loss��>


accuracy_1)\?&���\       ����	���)_��A{*O

prediction_loss��>

reg_loss�x<


total_lossVE�>


accuracy_1q=
?ǜ1�\       ����	�)_��A|*O

prediction_loss��?

reg_loss�x<


total_loss�y?


accuracy_1���>�3�}\       ����	�>�)_��A}*O

prediction_loss��>

reg_loss�x<


total_lossUE�>


accuracy_1q=
?3�u\       ����	=`�)_��A~*O

prediction_loss���>

reg_loss�x<


total_lossł�>


accuracy_1�?����\       ����	5z�)_��A*O

prediction_loss)\?

reg_loss�x<


total_lossD<?


accuracy_1�G�>\���]       a[��	���)_��A�*O

prediction_loss)\?

reg_loss�x<


total_lossD<?


accuracy_1�G�>0�w�]       a[��	��)_��A�*O

prediction_loss�z?

reg_loss�x<


total_loss�Z?


accuracy_1=
�>���]       a[��	���)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?UX�]       a[��	���)_��A�*O

prediction_loss�z?

reg_loss�x<


total_loss�Z?


accuracy_1=
�>��dh]       a[��	���)_��A�*O

prediction_loss�?

reg_loss�x<


total_loss��?


accuracy_1���>l=C]       a[��	(�)_��A�*O

prediction_loss\��>

reg_losswx<


total_loss�O�>


accuracy_1R�?Z�U�]       a[��	�'�)_��A�*O

prediction_loss�?

reg_losslx<


total_loss��?


accuracy_1���>�B�]       a[��	�B�)_��A�*O

prediction_loss�?

reg_loss`x<


total_loss��?


accuracy_1���>�ބ�]       a[��	N^�)_��A�*O

prediction_loss{�>

reg_lossVx<


total_loss�Ե>


accuracy_1��(?� �q]       a[��	>v�)_��A�*O

prediction_loss   ?

reg_lossJx<


total_loss�?


accuracy_1   ?A?��]       a[��	���)_��A�*O

prediction_loss���>

reg_lossAx<


total_loss���>


accuracy_1�?���]       a[��	��)_��A�*O

prediction_loss   ?

reg_loss6x<


total_loss�?


accuracy_1   ?5��]       a[��	ӿ�)_��A�*O

prediction_loss��>

reg_loss-x<


total_lossPE�>


accuracy_1q=
?���]       a[��	'��)_��A�*O

prediction_loss�z?

reg_loss!x<


total_loss�Z?


accuracy_1=
�>�G�]       a[��	V��)_��A�*O

prediction_loss)\?

reg_lossx<


total_lossA<?


accuracy_1�G�>�0�]       a[��	��)_��A�*O

prediction_loss\��>

reg_lossx<


total_loss�O�>


accuracy_1R�?���]       a[��	 (�)_��A�*O

prediction_loss   ?

reg_lossx<


total_loss�?


accuracy_1   ?y�o]       a[��	:@�)_��A�*O

prediction_lossq=
?

reg_loss�x<


total_loss�?


accuracy_1��>�Qf~]       a[��	[�)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?]��]       a[��	*r�)_��A�*O

prediction_loss�?

reg_loss�x<


total_loss��?


accuracy_1���>�r�]       a[��	+��)_��A�*O

prediction_loss���>

reg_loss�x<


total_loss���>


accuracy_1�?����]       a[��	ۣ�)_��A�*O

prediction_loss��>

reg_loss�x<


total_lossME�>


accuracy_1q=
?�&]       a[��	���)_��A�*O

prediction_loss�z?

reg_loss�x<


total_loss�Z?


accuracy_1=
�>�G]       a[��	���)_��A�*O

prediction_loss�?

reg_loss�x<


total_loss��?


accuracy_1���>T�u�]       a[��	���)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?tm]       a[��	��)_��A�*O

prediction_loss)\?

reg_loss�x<


total_loss@<?


accuracy_1�G�>X��]       a[��	\�)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?�B�]       a[��	i5�)_��A�*O

prediction_loss�z?

reg_loss�x<


total_loss�Z?


accuracy_1=
�>/�]       a[��	P�)_��A�*O

prediction_loss�z?

reg_loss�x<


total_loss�Z?


accuracy_1=
�>���2]       a[��	k�)_��A�*O

prediction_loss   ?

reg_losszx<


total_loss�?


accuracy_1   ?�|P]       a[��	��)_��A�*O

prediction_loss�?

reg_losspx<


total_loss��?


accuracy_1���>Uwj%]       a[��	ݙ�)_��A�*O

prediction_loss�G�>

reg_lossfx<


total_loss��>


accuracy_1)\?�n
X]       a[��	���)_��A�*O

prediction_lossq=
?

reg_loss[x<


total_loss�?


accuracy_1��>@.�]       a[��	���)_��A�*O

prediction_loss=
�>

reg_lossPx<


total_lossh��>


accuracy_1�z?}]y]       a[��	���)_��A�*O

prediction_loss=
�>

reg_lossIx<


total_lossg��>


accuracy_1�z?�Ʋ�]       a[��	��)_��A�*O

prediction_loss�z?

reg_loss<x<


total_loss�Z?


accuracy_1=
�>�ͥ]       a[��	��)_��A�*O

prediction_lossq=
?

reg_loss3x<


total_loss�?


accuracy_1��>|��&]       a[��	/�)_��A�*O

prediction_loss)\?

reg_loss&x<


total_loss><?


accuracy_1�G�>�Cqu]       a[��	0G�)_��A�*O

prediction_loss���>

reg_lossx<


total_loss���>


accuracy_1�?x�b�]       a[��	�`�)_��A�*O

prediction_loss��>

reg_lossx<


total_lossHE�>


accuracy_1q=
?@���]       a[��	ty�)_��A�*O

prediction_loss�?

reg_lossx<


total_loss��?


accuracy_1���>Fy��]       a[��	��)_��A�*O

prediction_loss�G�>

reg_loss�x<


total_loss��>


accuracy_1)\?�<)]       a[��	���)_��A�*O

prediction_loss��(?

reg_loss�x<


total_loss��,?


accuracy_1{�>��]       a[��	I��)_��A�*O

prediction_loss
�#?

reg_loss�x<


total_loss�'?


accuracy_1�Q�>h�t]       a[��	���)_��A�*O

prediction_loss)\?

reg_loss�x<


total_loss<<?


accuracy_1�G�>m�b-]       a[��	N��)_��A�*O

prediction_lossq=
?

reg_loss�x<


total_loss�?


accuracy_1��>#|��]       a[��	d�)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?1*{]       a[��	C�)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?�%�]       a[��	+5�)_��A�*O

prediction_loss=
�>

reg_loss�x<


total_lossc��>


accuracy_1�z?�h�;]       a[��	�L�)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?���]       a[��	kd�)_��A�*O

prediction_loss�z?

reg_loss�x<


total_loss�Z?


accuracy_1=
�>�N(]       a[��	=��)_��A�*O

prediction_loss��>

reg_loss�x<


total_lossDE�>


accuracy_1q=
?-��9]       a[��	|��)_��A�*O

prediction_loss�?

reg_loss�x<


total_loss��?


accuracy_1���>X�ޣ]       a[��	��)_��A�*O

prediction_loss�z?

reg_loss�x<


total_loss�Z?


accuracy_1=
�>�.��]       a[��	���)_��A�*O

prediction_loss\��>

reg_lossvx<


total_loss�O�>


accuracy_1R�?��D]       a[��	. �)_��A�*O

prediction_loss)\?

reg_losskx<


total_loss;<?


accuracy_1�G�>��O�]       a[��	a�)_��A�*O

prediction_lossq=
?

reg_loss`x<


total_loss�?


accuracy_1��>�쳑]       a[��	:;�)_��A�*O

prediction_loss)\?

reg_lossWx<


total_loss:<?


accuracy_1�G�>�ؓs]       a[��	�[�)_��A�*O

prediction_loss���>

reg_lossMx<


total_loss���>


accuracy_1�?� ��]       a[��	T��)_��A�*O

prediction_lossq=
?

reg_lossDx<


total_loss�?


accuracy_1��>S�{7]       a[��	��)_��A�*O

prediction_loss\��>

reg_loss7x<


total_loss~O�>


accuracy_1R�?��lN]       a[��	���)_��A�*O

prediction_loss���>

reg_loss-x<


total_loss���>


accuracy_1�?�}a�]       a[��	���)_��A�*O

prediction_loss)\?

reg_loss!x<


total_loss:<?


accuracy_1�G�>��
r]       a[��	��)_��A�*O

prediction_loss�G�>

reg_lossx<


total_loss��>


accuracy_1)\?l@4�]       a[��	gG�)_��A�*O

prediction_loss   ?

reg_lossx<


total_loss�?


accuracy_1   ?��=]       a[��	�q�)_��A�*O

prediction_lossq=
?

reg_lossx<


total_loss�?


accuracy_1��>bi7�]       a[��	R��)_��A�*O

prediction_loss)\?

reg_loss�x<


total_loss9<?


accuracy_1�G�>���I]       a[��	���)_��A�*O

prediction_loss���>

reg_loss�x<


total_loss���>


accuracy_1�?����]       a[��	��)_��A�*O

prediction_loss��>

reg_loss�x<


total_loss>E�>


accuracy_1q=
?�}2]       a[��	E,�)_��A�*O

prediction_loss���>

reg_loss�x<


total_loss���>


accuracy_1�?F��0]       a[��	B_�)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?_:�A]       a[��	��)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?�3Ϊ]       a[��	r��)_��A�*O

prediction_loss�?

reg_loss�x<


total_loss��?


accuracy_1���>tQ�z]       a[��	pB�)_��A�*O

prediction_loss��>

reg_loss�x<


total_loss<E�>


accuracy_1q=
?r�j�]       a[��	���)_��A�*O

prediction_loss�?

reg_loss�x<


total_loss��?


accuracy_1���>�1�I]       a[��	���)_��A�*O

prediction_loss)\?

reg_loss�x<


total_loss7<?


accuracy_1�G�>���]       a[��	���)_��A�*O

prediction_loss���>

reg_loss�x<


total_loss���>


accuracy_1�?�@��]       a[��	7�)_��A�*O

prediction_loss��>

reg_loss~x<


total_loss;E�>


accuracy_1q=
?����]       a[��	<K�)_��A�*O

prediction_loss)\?

reg_losssx<


total_loss7<?


accuracy_1�G�>]7,]       a[��	�v�)_��A�*O

prediction_loss��>

reg_losskx<


total_loss:E�>


accuracy_1q=
?��\]       a[��	}��)_��A�*O

prediction_lossq=
?

reg_loss`x<


total_loss~?


accuracy_1��>��G�]       a[��	���)_��A�*O

prediction_loss�z?

reg_lossVx<


total_loss�Z?


accuracy_1=
�>�g��]       a[��	��)_��A�*O

prediction_loss�G�>

reg_lossKx<


total_loss��>


accuracy_1)\?�6x,]       a[��	���)_��A�*O

prediction_loss��>

reg_lossBx<


total_loss9E�>


accuracy_1q=
?����]       a[��	�t�)_��A�*O

prediction_loss=
�>

reg_loss7x<


total_lossW��>


accuracy_1�z?2F]       a[��	ѯ�)_��A�*O

prediction_loss�?

reg_loss,x<


total_loss��?


accuracy_1���>̞<w]       a[��	���)_��A�*O

prediction_loss�?

reg_loss"x<


total_loss��?


accuracy_1���>��l�]       a[��	��)_��A�*O

prediction_loss)\?

reg_lossx<


total_loss5<?


accuracy_1�G�>Pv�]       a[��	�-�)_��A�*O

prediction_loss�?

reg_lossx<


total_loss��?


accuracy_1���>	�_�]       a[��	eR�)_��A�*O

prediction_loss���>

reg_lossx<


total_loss���>


accuracy_1�?�v��]       a[��	X�)_��A�*O

prediction_loss\��>

reg_loss�x<


total_losstO�>


accuracy_1R�?E)ظ]       a[��	�#�)_��A�*O

prediction_lossq=
?

reg_loss�x<


total_loss}?


accuracy_1��>���]       a[��	sg�)_��A�*O

prediction_loss�G�>

reg_loss�x<


total_loss��>


accuracy_1)\?�#s]       a[��	���)_��A�*O

prediction_loss�Q�>

reg_loss�x<


total_loss�>


accuracy_1
�#?�,3]       a[��	{��)_��A�*O

prediction_loss���>

reg_loss�x<


total_loss��>


accuracy_1��?���]       a[��	��)_��A�*O

prediction_loss��?

reg_loss�x<


total_loss�y?


accuracy_1���>G؅`]       a[��	*�)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?+�ڋ]       a[��	}X�)_��A�*O

prediction_loss�?

reg_loss�x<


total_loss��?


accuracy_1���>	��E]       a[��	���)_��A�*O

prediction_loss�?

reg_loss�x<


total_loss��?


accuracy_1���>�<�]       a[��	���)_��A�*O

prediction_lossq=
?

reg_loss�x<


total_loss{?


accuracy_1��>�Ό�]       a[��	Y��)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss
�?


accuracy_1   ?NEj]       a[��	��)_��A�*O

prediction_loss�?

reg_loss�x<


total_loss��?


accuracy_1���>D}e�]       a[��	?;�)_��A�*O

prediction_loss   ?

reg_loss~x<


total_loss
�?


accuracy_1   ?׉�;]       a[��	F\�)_��A�*O

prediction_loss���>

reg_losssx<


total_loss���>


accuracy_1�?�T��]       a[��	���)_��A�*O

prediction_loss���>

reg_losshx<


total_loss���>


accuracy_1�?�~.
]       a[��	W��)_��A�*O

prediction_loss=
�>

reg_loss_x<


total_lossP��>


accuracy_1�z?�H]       a[��	���)_��A�*O

prediction_loss=
�>

reg_lossVx<


total_lossP��>


accuracy_1�z?����]       a[��	���)_��A�*O

prediction_lossq=
?

reg_lossLx<


total_lossz?


accuracy_1��>n���]       a[��	��)_��A�*O

prediction_loss=
�>

reg_lossCx<


total_lossO��>


accuracy_1�z?��U]       a[��	�$�)_��A�*O

prediction_loss�G�>

reg_loss8x<


total_loss��>


accuracy_1)\?��[4]       a[��	�C�)_��A�*O

prediction_lossq=
?

reg_loss.x<


total_lossz?


accuracy_1��>�L�?]       a[��	�`�)_��A�*O

prediction_lossq=
?

reg_loss!x<


total_lossz?


accuracy_1��>o~�]       a[��	�|�)_��A�*O

prediction_lossq=
?

reg_lossx<


total_lossy?


accuracy_1��>�xh]       a[��	���)_��A�*O

prediction_loss=
�>

reg_lossx<


total_lossM��>


accuracy_1�z?�u��]       a[��	_��)_��A�*O

prediction_loss=
�>

reg_lossx<


total_lossM��>


accuracy_1�z?n�;]       a[��	���)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?�Y�3]       a[��	��)_��A�*O

prediction_lossq=
?

reg_loss�x<


total_lossy?


accuracy_1��>j&��]       a[��	�%�)_��A�*O

prediction_loss��?

reg_loss�x<


total_loss�y?


accuracy_1���>�P�]       a[��	O?�)_��A�*O

prediction_loss���>

reg_loss�x<


total_loss���>


accuracy_1�?h��D]       a[��	�^�)_��A�*O

prediction_loss\��>

reg_loss�x<


total_lossjO�>


accuracy_1R�?p9lu]       a[��	���)_��A�*O

prediction_loss���>

reg_loss�x<


total_loss���>


accuracy_1�?�3|]       a[��	u��)_��A�*O

prediction_loss�G�>

reg_loss�x<


total_loss��>


accuracy_1)\?߫0]       a[��	��)_��A�*O

prediction_loss���>

reg_loss�x<


total_loss���>


accuracy_1�?��$Q]       a[��	0��)_��A�*O

prediction_loss���>

reg_loss�x<


total_loss���>


accuracy_1�?By	]       a[��	�+�)_��A�*O

prediction_loss���>

reg_loss�x<


total_loss���>


accuracy_1�?M�^�]       a[��	�K�)_��A�*O

prediction_loss=
�>

reg_loss�x<


total_lossI��>


accuracy_1�z?��]       a[��	�j�)_��A�*O

prediction_loss   ?

reg_loss�x<


total_loss�?


accuracy_1   ?I+�h]       a[��	@��)_��A�*O

prediction_loss�Q�>

reg_losszx<


total_loss��>


accuracy_1
�#?H�|�]       a[��	I��)_��A�*O

prediction_loss���>

reg_lossqx<


total_lossٌ�>


accuracy_1��?�9�v]       a[��	b��)_��A�*O

prediction_loss)\�>

reg_lossgx<


total_loss4�>


accuracy_1�Q8?)��K]       a[��	���)_��A�*O

prediction_loss�G�>

reg_loss\x<


total_loss��>


accuracy_1)\?Ǜ|e]       a[��	���)_��A�*O

prediction_loss�G�>

reg_lossQx<


total_loss��>


accuracy_1)\?�?*�]       a[��	
�)_��A�*O

prediction_loss���>

reg_lossFx<


total_loss���>


accuracy_1�?7�f�]       a[��	#�)_��A�*O

prediction_loss�z?

reg_loss<x<


total_loss�Z?


accuracy_1=
�>��]       a[��	�;�)_��A�*O

prediction_loss�?

reg_loss1x<


total_loss��?


accuracy_1���>�Q�]       a[��	W�)_��A�*O

prediction_loss���>

reg_loss(x<


total_loss���>


accuracy_1�?���]       a[��	mp�)_��A�*O

prediction_loss���>

reg_lossx<


total_loss֌�>


accuracy_1��?�c<]       a[��	���)_��A�*O

prediction_loss��>

reg_lossx<


total_loss(E�>


accuracy_1q=
?>-�]       a[��	���)_��A�*O

prediction_loss�?

reg_loss
x<


total_loss��?


accuracy_1���>B��o]       a[��	���)_��A�*O

prediction_loss   ?

reg_loss x<


total_loss�?


accuracy_1   ?V4��]       a[��	���)_��A�*O

prediction_loss{.?

reg_loss� x<


total_loss�1?


accuracy_1
ף>���]       a[��	��)_��A�*O

prediction_loss   ?

reg_loss� x<


total_loss�?


accuracy_1   ?��']       a[��	�&�)_��A�*O

prediction_loss���>

reg_loss� x<


total_loss���>


accuracy_1�?C�}�]       a[��	@�)_��A�*O

prediction_loss�?

reg_loss� x<


total_loss��?


accuracy_1���>f��A]       a[��	�[�)_��A�*O

prediction_loss��?

reg_loss� x<


total_loss�y?


accuracy_1���>�̚]       a[��	$|�)_��A�*O

prediction_loss��>

reg_loss� x<


total_loss%E�>


accuracy_1q=
?j*�]       a[��	���)_��A�*O

prediction_lossR�?

reg_loss� x<


total_lossU�"?


accuracy_1\��>^d�]       a[��	y��)_��A�*O

prediction_loss
�#?

reg_loss� x<


total_loss�'?


accuracy_1�Q�>���]       a[��	���)_��A�*O

prediction_loss���>

reg_loss� x<


total_lossҌ�>


accuracy_1��?��At]       a[��	���)_��A�*O

prediction_loss���>

reg_loss� x<


total_loss�Y�>


accuracy_1333?�D�F]       a[��	��)_��A�*O

prediction_loss��>

reg_loss� x<


total_loss#E�>


accuracy_1q=
?Hi[]       a[��	�&�)_��A�*O

prediction_loss���>

reg_loss� x<


total_loss���>


accuracy_1�?����]       a[��	�A�)_��A�*O

prediction_loss{�>

reg_lossz x<


total_lossԵ>


accuracy_1��(?���]       a[��	�Z�)_��A�*O

prediction_lossq=
?

reg_lossq x<


total_losss?


accuracy_1��>A���]       a[��	�s�)_��A�*O

prediction_loss   ?

reg_lossh x<


total_loss�?


accuracy_1   ?�f-h]       a[��	���)_��A�*O

prediction_loss
�#?

reg_loss] x<


total_loss�'?


accuracy_1�Q�>�=\w]       a[��	��)_��A�*O

prediction_loss�G�>

reg_lossR x<


total_loss��>


accuracy_1)\?j���]       a[��	A��)_��A�*O

prediction_lossq=
?

reg_lossH x<


total_lossr?


accuracy_1��>IrC$]       a[��	���)_��A�*O

prediction_loss)\?

reg_loss= x<


total_loss*<?


accuracy_1�G�>,�}�]       a[��	9��)_��A�*O

prediction_lossq=
?

reg_loss4 x<


total_lossr?


accuracy_1��>�L��]       a[��	� *_��A�*O

prediction_loss�?

reg_loss) x<


total_loss��?


accuracy_1���>3��]       a[��	�# *_��A�*O

prediction_loss   ?

reg_loss x<


total_loss �?


accuracy_1   ?j�{u]       a[��	6; *_��A�*O

prediction_loss��>

reg_loss x<


total_loss E�>


accuracy_1q=
?�`�]       a[��	~Q *_��A�*O

prediction_lossR�?

reg_loss	 x<


total_lossR�"?


accuracy_1\��>m��O]       a[��	�h *_��A�*O

prediction_loss\��>

reg_loss  x<


total_loss\O�>


accuracy_1R�?B��&]       a[��	� *_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossR�"?


accuracy_1\��>mV�]       a[��	B� *_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss<��>


accuracy_1�z?2�
�]       a[��	}� *_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�d��]       a[��	�� *_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>���]       a[��	�� *_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss;��>


accuracy_1�z?I��h]       a[��	�� *_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss(<?


accuracy_1�G�>�p+]       a[��	'*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?$��]       a[��	$**_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?���]       a[��	ux*_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossP�"?


accuracy_1\��>6�{r]       a[��	:�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>����]       a[��	h�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_losso?


accuracy_1��>�W��]       a[��	��*_��A�*O

prediction_loss
�#?

reg_lossy�w<


total_loss�'?


accuracy_1�Q�>�! �]       a[��	$�*_��A�*O

prediction_loss���>

reg_lossn�w<


total_loss���>


accuracy_1�?��j]       a[��	�	*_��A�*O

prediction_lossq=
?

reg_losse�w<


total_losso?


accuracy_1��>�'�]       a[��	�!*_��A�*O

prediction_loss\��>

reg_loss[�w<


total_lossWO�>


accuracy_1R�?��W]       a[��	�:*_��A�*O

prediction_loss���>

reg_lossQ�w<


total_loss���>


accuracy_1�?s�T6]       a[��	�S*_��A�*O

prediction_loss�?

reg_lossG�w<


total_loss��?


accuracy_1���>r�E]       a[��	Ql*_��A�*O

prediction_loss�?

reg_loss;�w<


total_loss��?


accuracy_1���>��er]       a[��	�*_��A�*O

prediction_loss��>

reg_loss2�w<


total_lossE�>


accuracy_1q=
?�#$1]       a[��	��*_��A�*O

prediction_loss���>

reg_loss'�w<


total_lossƌ�>


accuracy_1��?5"�]       a[��	)�*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?Gߜ]       a[��	u�*_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss%<?


accuracy_1�G�>���v]       a[��	��*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_lossm?


accuracy_1��>��d�]       a[��	C�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss%<?


accuracy_1�G�>���\]       a[��	�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?S�h�]       a[��	�,*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�ʹ/]       a[��	�B*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?Uƽ�]       a[��	F]*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?>��]       a[��	�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>ڱ2 ]       a[��	I�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>��t]       a[��	/�*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�y?


accuracy_1���>��9]       a[��	h*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�y?


accuracy_1���>����]       a[��	�V*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>���r]       a[��	�w*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�¥]       a[��	��*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?��7�]       a[��	�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossk?


accuracy_1��>��(,]       a[��	:�*_��A�*O

prediction_loss   ?

reg_lossv�w<


total_loss��?


accuracy_1   ?�$��]       a[��	u�*_��A�*O

prediction_loss\��>

reg_lossl�w<


total_lossOO�>


accuracy_1R�?�rr]       a[��	*_��A�*O

prediction_loss��>

reg_loss`�w<


total_lossE�>


accuracy_1q=
?I�]       a[��	)&*_��A�*O

prediction_loss   ?

reg_lossW�w<


total_loss��?


accuracy_1   ?H�5�]       a[��	�B*_��A�*O

prediction_loss�?

reg_lossL�w<


total_loss��?


accuracy_1���>CIJ]       a[��	O\*_��A�*O

prediction_loss   ?

reg_lossD�w<


total_loss��?


accuracy_1   ?�*u�]       a[��	.v*_��A�*O

prediction_loss
�#?

reg_loss8�w<


total_loss�'?


accuracy_1�Q�>�޲]       a[��	\�*_��A�*O

prediction_loss\��>

reg_loss.�w<


total_lossMO�>


accuracy_1R�?����]       a[��	P�*_��A�*O

prediction_loss��>

reg_loss$�w<


total_lossE�>


accuracy_1q=
?7E�]       a[��	j�*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?9
5S]       a[��	��*_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Z?


accuracy_1=
�>�O��]       a[��	��*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?A�B�]       a[��	�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>mL��]       a[��	�(*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss!<?


accuracy_1�G�>�}.]       a[��	-B*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss~��>


accuracy_1�?%.��]       a[��	�Y*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?%�D]       a[��	"q*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�5�M]       a[��	��*_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossI�"?


accuracy_1\��>��%�]       a[��	4�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�Y�>


accuracy_1333?��Q]       a[��	|�*_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossI�"?


accuracy_1\��>��@]       a[��	O�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?J9#�]       a[��	y�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss|��>


accuracy_1�?I�2�]       a[��	q*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�s*]       a[��		*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss{��>


accuracy_1�?�~2�]       a[��	�0*_��A�*O

prediction_loss   ?

reg_loss{�w<


total_loss��?


accuracy_1   ?"� ]       a[��	�F*_��A�*O

prediction_loss)\?

reg_lossp�w<


total_loss<?


accuracy_1�G�>B ��]       a[��	�a*_��A�*O

prediction_loss\��>

reg_lossg�w<


total_lossGO�>


accuracy_1R�?��d]       a[��	p{*_��A�*O

prediction_loss��>

reg_loss^�w<


total_loss
E�>


accuracy_1q=
?z�,]       a[��	͒*_��A�*O

prediction_loss   ?

reg_lossS�w<


total_loss��?


accuracy_1   ?;�]       a[��	 �*_��A�*O

prediction_loss�G�>

reg_lossI�w<


total_loss��>


accuracy_1)\?$�]       a[��	ۿ*_��A�*O

prediction_loss�?

reg_loss@�w<


total_loss��?


accuracy_1���>�H�W]       a[��	=�*_��A�*O

prediction_loss�G�>

reg_loss2�w<


total_loss��>


accuracy_1)\?d�]       a[��	_�*_��A�*O

prediction_loss���>

reg_loss*�w<


total_lossx��>


accuracy_1�?��]       a[��	J*_��A�*O

prediction_loss{.?

reg_loss�w<


total_losso�1?


accuracy_1
ף>�s�G]       a[��	S$*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>��wk]       a[��	C;*_��A�*O

prediction_loss���>

reg_loss�w<


total_lossw��>


accuracy_1�?�?2]       a[��	T*_��A�*O

prediction_loss\��>

reg_loss�w<


total_lossDO�>


accuracy_1R�?�eJ]       a[��	8k*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_losse?


accuracy_1��>��]       a[��	��*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>�WM]       a[��	5�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>�0�]       a[��	�*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossE�>


accuracy_1q=
?)��%]       a[��	F�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?j�]]       a[��	��*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?	O�]       a[��	�	*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossE�>


accuracy_1q=
?�C�]       a[��	�	*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossd?


accuracy_1��>p�ӵ]       a[��	�7	*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�h��]       a[��	�O	*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossc?


accuracy_1��>E:�v]       a[��	�h	*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?Ǧ�]       a[��	�	*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossE�>


accuracy_1q=
?��~�]       a[��	�	*_��A�*O

prediction_loss�G�>

reg_lossy�w<


total_loss��>


accuracy_1)\?Y��
]       a[��	ޯ	*_��A�*O

prediction_loss)\?

reg_lossp�w<


total_loss<?


accuracy_1�G�>���q]       a[��	P�	*_��A�*O

prediction_loss�?

reg_lossd�w<


total_loss��?


accuracy_1���>��#u]       a[��	��	*_��A�*O

prediction_loss�?

reg_loss[�w<


total_loss��?


accuracy_1���>���G]       a[��	D�	*_��A�*O

prediction_loss=
�>

reg_lossQ�w<


total_loss ��>


accuracy_1�z?��]       a[��	�
*_��A�*O

prediction_loss��>

reg_lossG�w<


total_lossE�>


accuracy_1q=
?K��]       a[��	�*
*_��A�*O

prediction_loss=
�>

reg_loss<�w<


total_loss��>


accuracy_1�z?�)h]]       a[��	 C
*_��A�*O

prediction_loss)\?

reg_loss2�w<


total_loss<?


accuracy_1�G�>��KL]       a[��	^
*_��A�*O

prediction_lossq=
?

reg_loss)�w<


total_lossb?


accuracy_1��>�r�]       a[��	2v
*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>���e]       a[��	*�
*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?і��]       a[��	/�
*_��A�*O

prediction_loss�G�>

reg_loss	�w<


total_loss��>


accuracy_1)\?��]       a[��	��
*_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossB�"?


accuracy_1\��>A�Ժ]       a[��	�
*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>پ�]       a[��	(�
*_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss;O�>


accuracy_1R�?�C�]       a[��	�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>5,?�]       a[��	b-*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>��;�]       a[��	�M*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss`?


accuracy_1��>��B~]       a[��	�e*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>/Dؼ]       a[��	1~*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>��]       a[��	��*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?0� ]       a[��	ڭ*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?�X;�]       a[��	��*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?޾��]       a[��	i*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?4�g�]       a[��	�1*_��A�*O

prediction_loss   ?

reg_lossw�w<


total_loss��?


accuracy_1   ?�ޱ]       a[��	N*_��A�*O

prediction_loss��?

reg_losso�w<


total_loss�y?


accuracy_1���>�n�F]       a[��	�f*_��A�*O

prediction_loss=
�>

reg_lossc�w<


total_loss��>


accuracy_1�z?����]       a[��	�~*_��A�*O

prediction_loss��>

reg_lossY�w<


total_loss�D�>


accuracy_1q=
?̷��]       a[��	��*_��A�*O

prediction_loss   ?

reg_lossN�w<


total_loss��?


accuracy_1   ?�9�]       a[��	p�*_��A�*O

prediction_loss�z?

reg_lossC�w<


total_loss�Z?


accuracy_1=
�>�Ѡr]       a[��	��*_��A�*O

prediction_lossq=
?

reg_loss8�w<


total_loss^?


accuracy_1��>��?]]       a[��	��*_��A�*O

prediction_loss�G�>

reg_loss0�w<


total_loss��>


accuracy_1)\?�M�|]       a[��	�*_��A�*O

prediction_loss�?

reg_loss#�w<


total_loss��?


accuracy_1���>�7�c]       a[��	�2*_��A�*O

prediction_loss���>

reg_loss�w<


total_lossh��>


accuracy_1�?���]       a[��	�[*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�D�>


accuracy_1q=
?[W!�]       a[��	 |*_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Z?


accuracy_1=
�>z��]       a[��	v�*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossg��>


accuracy_1�?s�S]       a[��	C�*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss��>


accuracy_1�z?��]       a[��	�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>m6c]       a[��	�V*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?��u�]       a[��	*�*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss��>


accuracy_1�z?�Ҿ�]       a[��	m�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>"��]       a[��	�'*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss��>


accuracy_1�z?o�]       a[��	��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss\?


accuracy_1��>T�]       a[��	��*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?���]       a[��	�#*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>#��]       a[��	�J*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?t!�,]       a[��	�v*_��A�*O

prediction_loss
ף>

reg_loss��w<


total_lossޖ�>


accuracy_1{.?�r9u]       a[��	M�*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?(�ĥ]       a[��	�*_��A�*O

prediction_loss�?

reg_lossu�w<


total_loss��?


accuracy_1���>�-�]       a[��	C*_��A�*O

prediction_lossq=
?

reg_lossi�w<


total_loss[?


accuracy_1��>�<i�]       a[��	�<*_��A�*O

prediction_loss��>

reg_loss`�w<


total_loss�D�>


accuracy_1q=
?W��H]       a[��	�b*_��A�*O

prediction_loss���>

reg_lossU�w<


total_lossb��>


accuracy_1�?�^�/]       a[��	��*_��A�*O

prediction_loss�?

reg_lossK�w<


total_loss��?


accuracy_1���>����]       a[��	F�*_��A�*O

prediction_loss   ?

reg_loss@�w<


total_loss��?


accuracy_1   ?r({�]       a[��	�
*_��A�*O

prediction_loss=
�>

reg_loss6�w<


total_loss��>


accuracy_1�z?�Y|]       a[��	�3*_��A�*O

prediction_loss�G�>

reg_loss,�w<


total_loss�>


accuracy_1)\?U	�5]       a[��	uX*_��A�*O

prediction_lossq=
?

reg_loss!�w<


total_lossZ?


accuracy_1��>�/_g]       a[��	=~*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_lossY?


accuracy_1��>��]       a[��	j�*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�D�>


accuracy_1q=
?��&]       a[��	��*_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss��>


accuracy_1�z?Јj]       a[��	�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>wG��]       a[��	&9*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>�)�]       a[��	�Z*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss^��>


accuracy_1�?�=8H]       a[��	_*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss��>


accuracy_1�z?n�]       a[��	;�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss^��>


accuracy_1�?���g]       a[��	J�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?��]       a[��	\*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>��]       a[��	�D*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>���F]       a[��	ak*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?D��u]       a[��	�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?G�]       a[��	p�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�,3T]       a[��	2�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss[��>


accuracy_1�?�p�E]       a[��	5
*_��A�*O

prediction_loss��>

reg_loss~�w<


total_loss�D�>


accuracy_1q=
?��]       a[��	�6*_��A�*O

prediction_loss�z?

reg_lossu�w<


total_loss�Z?


accuracy_1=
�>�`R$]       a[��	zR*_��A�*O

prediction_loss�Q�>

reg_lossf�w<


total_loss��>


accuracy_1
�#?g�<]       a[��	o*_��A�*O

prediction_loss   ?

reg_loss^�w<


total_loss��?


accuracy_1   ?} �]       a[��	v�*_��A�*O

prediction_loss���>

reg_lossT�w<


total_loss���>


accuracy_1��?�H[.]       a[��	P�*_��A�*O

prediction_loss   ?

reg_lossI�w<


total_loss��?


accuracy_1   ?�X6]       a[��	H�*_��A�*O

prediction_loss)\?

reg_loss>�w<


total_loss<?


accuracy_1�G�>��j�]       a[��	~�*_��A�*O

prediction_loss   ?

reg_loss5�w<


total_loss��?


accuracy_1   ?}"�E]       a[��	U�*_��A�*O

prediction_loss��>

reg_loss+�w<


total_loss�D�>


accuracy_1q=
?���]       a[��	e*_��A�*O

prediction_lossq=
?

reg_loss �w<


total_lossV?


accuracy_1��>��/]       a[��	�2*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?��w]       a[��	nQ*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?
�=]       a[��	�k*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�D�>


accuracy_1q=
?A�]       a[��	ˆ*_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss$O�>


accuracy_1R�?d��]       a[��	V�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>��(l]       a[��	Ǹ*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>��3�]       a[��	9�*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss}y?


accuracy_1���>���]       a[��	��*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>�.�G]       a[��	-*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>���]       a[��	�!*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_losst�>


accuracy_1)\?X�s�]       a[��	D6*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossU��>


accuracy_1�?u�[]       a[��	�N*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?!�2�]       a[��	Ui*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?`�?u]       a[��	�*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss��>


accuracy_1�z?q�8u]       a[��	��*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossr�>


accuracy_1)\?��m�]       a[��	g�*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss|y?


accuracy_1���>�?�]       a[��	=�*_��A�*O

prediction_loss�G�>

reg_lossv�w<


total_lossr�>


accuracy_1)\?�e{x]       a[��	��*_��A�*O

prediction_loss   ?

reg_lossj�w<


total_loss��?


accuracy_1   ?,*h)]       a[��	 *_��A�*O

prediction_lossq=
?

reg_lossa�w<


total_lossS?


accuracy_1��>��EO]       a[��	 *_��A�*O

prediction_loss   ?

reg_lossU�w<


total_loss��?


accuracy_1   ?���]       a[��	'2*_��A�*O

prediction_loss{.?

reg_lossI�w<


total_loss\�1?


accuracy_1
ף>��|]       a[��	�H*_��A�*O

prediction_loss=
�>

reg_loss?�w<


total_loss���>


accuracy_1�z??�	]       a[��	[a*_��A�*O

prediction_loss���>

reg_loss6�w<


total_lossQ��>


accuracy_1�?�-+$]       a[��	�{*_��A�*O

prediction_loss�?

reg_loss*�w<


total_loss��?


accuracy_1���>���b]       a[��	�*_��A�*O

prediction_lossq=
?

reg_loss!�w<


total_lossR?


accuracy_1��>���G]       a[��	i�*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>��J]       a[��	��*_��A�*O

prediction_loss��?

reg_loss�w<


total_losszy?


accuracy_1���>υ�J]       a[��	s�*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>���]       a[��	��*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>;��{]       a[��	�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossQ?


accuracy_1��>D4]       a[��	�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?�n]       a[��	:*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>)�Ǘ]       a[��	�Q*_��A�*O

prediction_loss��?

reg_loss��w<


total_lossyy?


accuracy_1���>$ Z�]       a[��	+j*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossM��>


accuracy_1�?s��]       a[��	ǁ*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossP?


accuracy_1��>�{J�]       a[��	��*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossL��>


accuracy_1�?�\�]       a[��	��*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossk�>


accuracy_1)\?u
�]       a[��	��*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>K!]       a[��	V�*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossK��>


accuracy_1�?���]       a[��	g*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>�	l�]       a[��	(b*_��A�*O

prediction_loss���>

reg_lossq�w<


total_lossK��>


accuracy_1�?8�]       a[��	{*_��A�*O

prediction_lossq=
?

reg_lossd�w<


total_lossO?


accuracy_1��>�v-]       a[��	��*_��A�*O

prediction_loss   ?

reg_loss[�w<


total_loss��?


accuracy_1   ?��W]       a[��	�*_��A�*O

prediction_loss���>

reg_lossR�w<


total_lossJ��>


accuracy_1�?e��G]       a[��	��*_��A�*O

prediction_loss��>

reg_lossH�w<


total_loss�D�>


accuracy_1q=
?m��]       a[��	k�*_��A�*O

prediction_loss���>

reg_loss<�w<


total_lossI��>


accuracy_1�?l�$�]       a[��	� *_��A�*O

prediction_loss
�#?

reg_loss2�w<


total_loss�'?


accuracy_1�Q�>�LN�]       a[��	�<*_��A�*O

prediction_loss=
�>

reg_loss&�w<


total_loss���>


accuracy_1�z?��f7]       a[��	 X*_��A�*O

prediction_loss
�#?

reg_loss�w<


total_loss�'?


accuracy_1�Q�>$�]       a[��	�s*_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Z?


accuracy_1=
�>���]       a[��	E�*_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?��Ac]       a[��	�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?�m�]       a[��	��*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?1� I]       a[��	a*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>l���]       a[��	e4*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?U�<�]       a[��	iR*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�K<]       a[��	�{*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�G��]       a[��	@�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossd�>


accuracy_1)\?�]       a[��	v�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss<?


accuracy_1�G�>���
]       a[��	��*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>�Ute]       a[��	-*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?Eo]       a[��	�<*_��A�*O

prediction_loss��?

reg_loss��w<


total_lossty?


accuracy_1���>a�K]       a[��	W*_��A�*O

prediction_loss��?

reg_loss��w<


total_lossty?


accuracy_1���>�L�A]       a[��	�o*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossC��>


accuracy_1�?Z�]       a[��	��*_��A�*O

prediction_lossq=
?

reg_lossw�w<


total_lossK?


accuracy_1��>����]       a[��	B�*_��A�*O

prediction_loss��>

reg_lossm�w<


total_loss�D�>


accuracy_1q=
?�g9�]       a[��	(�*_��A�*O

prediction_loss�G�>

reg_lossc�w<


total_lossa�>


accuracy_1)\?����]       a[��	5*_��A�*O

prediction_loss���>

reg_lossX�w<


total_lossB��>


accuracy_1�?1�9]       a[��	�*_��A�*O

prediction_loss
�#?

reg_lossM�w<


total_loss�'?


accuracy_1�Q�>;)�l]       a[��	�4*_��A�*O

prediction_loss)\?

reg_lossB�w<


total_loss<?


accuracy_1�G�>i���]       a[��	{J*_��A�*O

prediction_loss=
�>

reg_loss:�w<


total_loss���>


accuracy_1�z?礸�]       a[��	�a*_��A�*O

prediction_loss�?

reg_loss.�w<


total_loss��?


accuracy_1���>"\]       a[��	�z*_��A�*O

prediction_loss��>

reg_loss#�w<


total_loss�D�>


accuracy_1q=
?W��H]       a[��	��*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�D�>


accuracy_1q=
?o��w]       a[��	Ω*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?�T�5]       a[��	k�*_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?]	I�]       a[��	��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossI?


accuracy_1��>iҵ�]       a[��	�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?uG��]       a[��	3*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>U��]       a[��	�8*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss]�>


accuracy_1)\?¸>�]       a[��	�P*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>��i]       a[��	bi*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss\�>


accuracy_1)\?�s�i]       a[��	
�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>�V�]       a[��	W�*_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss
O�>


accuracy_1R�?�t��]       a[��	��*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?�B"W]       a[��	"�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss<��>


accuracy_1�?�)X�]       a[��	 �*_��A�*O

prediction_loss
�#?

reg_loss��w<


total_loss�'?


accuracy_1�Q�>	Ţ]       a[��	��*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossZ�>


accuracy_1)\?Os8]       a[��	� *_��A�*O

prediction_lossq=
?

reg_loss~�w<


total_lossG?


accuracy_1��>��t�]       a[��	�$ *_��A�*O

prediction_loss)\?

reg_lossv�w<


total_loss�;?


accuracy_1�G�>{�u�]       a[��	�7 *_��A�*O

prediction_loss���>

reg_lossk�w<


total_loss:��>


accuracy_1�?A�k�]       a[��	@M *_��A�*O

prediction_loss���>

reg_lossa�w<


total_lossx��>


accuracy_1��?zW�]       a[��	If *_��A�*O

prediction_loss�?

reg_lossW�w<


total_loss��?


accuracy_1���>6��C]       a[��	�{ *_��A�*O

prediction_loss�z?

reg_lossN�w<


total_loss�Z?


accuracy_1=
�>��D�]       a[��	�� *_��A�*O

prediction_loss���>

reg_lossC�w<


total_loss9��>


accuracy_1�?�01]       a[��	n� *_��A�*O

prediction_loss�z?

reg_loss8�w<


total_loss�Z?


accuracy_1=
�>ֈ,]       a[��	+� *_��A�*O

prediction_loss�G�>

reg_loss.�w<


total_lossW�>


accuracy_1)\?��`]       a[��	4� *_��A�*O

prediction_loss�?

reg_loss#�w<


total_loss��?


accuracy_1���>��]       a[��	$� *_��A�*O

prediction_loss���>

reg_loss�w<


total_lossv��>


accuracy_1��?����]       a[��	6!*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�D�>


accuracy_1q=
?Ԭqb]       a[��	�!*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?N�]       a[��	#/!*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>��\]       a[��	5D!*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�6j�]       a[��	�f!*_��A�*O

prediction_loss���>

reg_loss��w<


total_losst��>


accuracy_1��?I� K]       a[��	�}!*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>n�T*]       a[��	�!*_��A�*O

prediction_loss)\�>

reg_loss��w<


total_loss��>


accuracy_1�Q8?�5��]       a[��	3�!*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?��ŭ]       a[��	8�!*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?�/�]       a[��	��!*_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss%�"?


accuracy_1\��>��4]       a[��	��!*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�mF]       a[��	 "*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>��n]       a[��	D"*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossR�>


accuracy_1)\?��]       a[��	�,"*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossC?


accuracy_1��>�>|]       a[��	�A"*_��A�*O

prediction_lossq=
?

reg_loss{�w<


total_lossC?


accuracy_1��>=�Q]       a[��	�X"*_��A�*O

prediction_loss���>

reg_losso�w<


total_lossp��>


accuracy_1��?����]       a[��	�p"*_��A�*O

prediction_lossq=
?

reg_losse�w<


total_lossC?


accuracy_1��>]H��]       a[��	�"*_��A�*O

prediction_loss�G�>

reg_loss^�w<


total_lossQ�>


accuracy_1)\?�6f*]       a[��	�"*_��A�*O

prediction_loss��>

reg_lossQ�w<


total_loss�D�>


accuracy_1q=
?Uc�]       a[��	��"*_��A�*O

prediction_loss��>

reg_lossG�w<


total_loss�D�>


accuracy_1q=
?6}�}]       a[��	��"*_��A�*O

prediction_lossq=
?

reg_loss<�w<


total_lossB?


accuracy_1��>A4B]       a[��	��"*_��A�*O

prediction_loss�G�>

reg_loss4�w<


total_lossP�>


accuracy_1)\?l��]       a[��	��"*_��A�*O

prediction_loss�?

reg_loss(�w<


total_loss��?


accuracy_1���>ߵ�]       a[��	#*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>�&ϯ]       a[��	�%#*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_lossA?


accuracy_1��>�Ȱ�]       a[��	�;#*_��A�*O

prediction_loss)\?

reg_loss
�w<


total_loss�;?


accuracy_1�G�>��]       a[��	�Q#*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossN�>


accuracy_1)\?����]       a[��	i#*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>A��{]       a[��	5~#*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?�O�]       a[��	ɓ#*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss.��>


accuracy_1�?�X�]       a[��	�#*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?4�f�]       a[��	��#*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossk��>


accuracy_1��?�-�]       a[��	M�#*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss@?


accuracy_1��>�eg�]       a[��	�#*_��A�*O

prediction_loss��(?

reg_loss��w<


total_loss��,?


accuracy_1{�>+�X�]       a[��	�$*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>0Wc�]       a[��	�$*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossj��>


accuracy_1��?[��E]       a[��	�,$*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�R�]       a[��	=C$*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss??


accuracy_1��>�]f]       a[��	SX$*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss+��>


accuracy_1�? )]       a[��	o$*_��A�*O

prediction_loss�G�>

reg_loss}�w<


total_lossJ�>


accuracy_1)\?��p]       a[��	��$*_��A�*O

prediction_loss��>

reg_lossr�w<


total_loss�D�>


accuracy_1q=
?����]       a[��	�$*_��A�*O

prediction_loss=
�>

reg_loss]�w<


total_loss���>


accuracy_1�z?��]       a[��	k�$*_��A�*O

prediction_loss)\?

reg_lossR�w<


total_loss�;?


accuracy_1�G�>�]       a[��	�%*_��A�*O

prediction_lossq=
?

reg_lossG�w<


total_loss>?


accuracy_1��>���]       a[��	%%*_��A�*O

prediction_loss���>

reg_loss<�w<


total_lossg��>


accuracy_1��?��]       a[��	�5%*_��A�*O

prediction_loss   ?

reg_loss5�w<


total_loss��?


accuracy_1   ?��]       a[��	�J%*_��A�*O

prediction_loss   ?

reg_loss)�w<


total_loss��?


accuracy_1   ?�Ey]       a[��	$a%*_��A�*O

prediction_loss���>

reg_loss �w<


total_loss(��>


accuracy_1�?��ӝ]       a[��	yy%*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?L��J]       a[��	�%*_��A�*O

prediction_loss��(?

reg_loss	�w<


total_loss��,?


accuracy_1{�>��^]       a[��	��%*_��A�*O

prediction_loss��?

reg_loss��w<


total_lossfy?


accuracy_1���>!V��]       a[��	��%*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?�UF]       a[��	*&*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossE�>


accuracy_1)\?Y�G\]       a[��	�&*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossE�>


accuracy_1)\?֨/�]       a[��	�2&*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss&��>


accuracy_1�?��d]       a[��	H&*_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�N�>


accuracy_1R�?�j]       a[��	�h&*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>�{|W]       a[��	��&*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossc��>


accuracy_1��?ؐ��]       a[��	��&*_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss�"?


accuracy_1\��>WD]       a[��	��&*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?�H�]       a[��	�'*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>��R]       a[��	�D'*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>'w2]       a[��	w'*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?
��q]       a[��	�((*_��A�*O

prediction_loss   ?

reg_lossw�w<


total_loss��?


accuracy_1   ?�Z�L]       a[��	�P(*_��A�*O

prediction_loss=
�>

reg_lossl�w<


total_loss���>


accuracy_1�z?�n�]       a[��	%z(*_��A�*O

prediction_loss���>

reg_lossc�w<


total_loss`��>


accuracy_1��?40��]       a[��	��(*_��A�*O

prediction_lossq=
?

reg_lossX�w<


total_loss:?


accuracy_1��>���X]       a[��	�)*_��A�*O

prediction_loss���>

reg_lossM�w<


total_loss!��>


accuracy_1�?�|6N]       a[��	j0)*_��A�*O

prediction_loss�?

reg_lossC�w<


total_loss��?


accuracy_1���>�Ȑ|]       a[��	qX)*_��A�*O

prediction_loss=
�>

reg_loss9�w<


total_loss���>


accuracy_1�z?>��]       a[��	Ӽ)*_��A�*O

prediction_loss���>

reg_loss/�w<


total_loss^��>


accuracy_1��?�HE]       a[��	{�)*_��A�*O

prediction_lossq=
?

reg_loss&�w<


total_loss:?


accuracy_1��>�Y$f]       a[��	��)*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss9?


accuracy_1��>�	��]       a[��	}!**_��A�*O

prediction_loss{�>

reg_loss�w<


total_lossԵ>


accuracy_1��(?��^]       a[��	�C**_��A�*O

prediction_loss���>

reg_loss	�w<


total_loss��>


accuracy_1�?0�,]       a[��	S�**_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss9?


accuracy_1��>!.�]]       a[��	��**_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss9?


accuracy_1��>V��]       a[��	��**_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?%~G�]       a[��	+*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss\��>


accuracy_1��?�D0�]       a[��	�+*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?Q��I]       a[��	_D,*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss8?


accuracy_1��>q3qm]       a[��	4g,*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?%R�]       a[��	��,*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?,Hx�]       a[��	ް,*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?DVZ�]       a[��	��,*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss7?


accuracy_1��>��3B]       a[��	-*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�gY%]       a[��	�2-*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?�j]       a[��	KY-*_��A�*O

prediction_loss\��>

reg_loss�w<


total_loss�N�>


accuracy_1R�?���]       a[��	 �-*_��A�*O

prediction_loss�G�>

reg_lossu�w<


total_loss:�>


accuracy_1)\?�,~�]       a[��	��-*_��A�*O

prediction_loss��>

reg_lossj�w<


total_loss�D�>


accuracy_1q=
?��N�]       a[��	��-*_��A�*O

prediction_loss�?

reg_lossa�w<


total_loss~�?


accuracy_1���>|�mu]       a[��	�.*_��A�*O

prediction_loss���>

reg_lossW�w<


total_loss��>


accuracy_1�?��G]       a[��	H6.*_��A�*O

prediction_loss��>

reg_lossK�w<


total_loss�D�>


accuracy_1q=
?���%]       a[��	U.*_��A�*O

prediction_loss)\?

reg_loss@�w<


total_loss�;?


accuracy_1�G�>;Z]]       a[��	-�.*_��A�*O

prediction_loss   ?

reg_loss6�w<


total_loss��?


accuracy_1   ?�h��]       a[��	@�.*_��A�*O

prediction_loss�z?

reg_loss*�w<


total_loss�Z?


accuracy_1=
�>E�R�]       a[��	�/*_��A�*O

prediction_loss��?

reg_loss �w<


total_loss^y?


accuracy_1���>^>8R]       a[��	�*/*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss|�?


accuracy_1���>��b�]       a[��	�I/*_��A�*O

prediction_loss   ?

reg_loss
�w<


total_loss��?


accuracy_1   ?� ]�]       a[��	�i/*_��A�*O

prediction_loss���>

reg_loss �w<


total_loss��>


accuracy_1�?r/P�]       a[��	��/*_��A�*O

prediction_loss{�>

reg_loss��w<


total_lossԵ>


accuracy_1��(?�i!]       a[��	�/*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?�I]       a[��	��/*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>��^']       a[��	�	0*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_losss�>


accuracy_1
�#?ݙ�]       a[��	�$0*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�|�G]       a[��	�A0*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?��Ј]       a[��	O\0*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z? wF�]       a[��	`v0*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss{�?


accuracy_1���>Q�W]       a[��	�0*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss]y?


accuracy_1���>�oD]       a[��	��0*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossz�?


accuracy_1���>sʾ}]       a[��	��0*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?:n��]       a[��	/�0*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossz�?


accuracy_1���>\�]       a[��	��0*_��A�*O

prediction_loss=
�>

reg_lossz�w<


total_loss���>


accuracy_1�z?��V]       a[��	Q1*_��A�*O

prediction_loss�G�>

reg_lossq�w<


total_loss2�>


accuracy_1)\?���]       a[��	L51*_��A�*O

prediction_loss�?

reg_lossh�w<


total_lossz�?


accuracy_1���>�k]       a[��	<N1*_��A�*O

prediction_loss=
�>

reg_loss]�w<


total_loss���>


accuracy_1�z?��]       a[��	�f1*_��A�*O

prediction_lossq=
?

reg_lossS�w<


total_loss2?


accuracy_1��>f�	]       a[��	|1*_��A�*O

prediction_loss�?

reg_lossJ�w<


total_lossy�?


accuracy_1���>�8�]       a[��	�1*_��A�*O

prediction_loss   ?

reg_loss?�w<


total_loss��?


accuracy_1   ?����]       a[��	?�1*_��A�*O

prediction_loss\��>

reg_loss4�w<


total_loss�N�>


accuracy_1R�?p�P�]       a[��	�1*_��A�*O

prediction_loss\��>

reg_loss+�w<


total_loss�N�>


accuracy_1R�?Q�v9]       a[��	��1*_��A�*O

prediction_loss�?

reg_loss"�w<


total_lossy�?


accuracy_1���>g�Lt]       a[��	x�1*_��A�*O

prediction_loss�?

reg_loss�w<


total_lossx�?


accuracy_1���>���]       a[��	F2*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss.�>


accuracy_1)\?�j�]       a[��	K2*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss1?


accuracy_1��>Lg�]       a[��	Q/2*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�=6/]       a[��	�?2*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�o�]       a[��	qV2*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossx�?


accuracy_1���>��C]       a[��	n2*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>�k!]       a[��	�2*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss,�>


accuracy_1)\?A��]       a[��	Օ2*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?U��]       a[��	Ҩ2*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>�(�]       a[��	��2*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�n]       a[��	_�2*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?s��]       a[��	��2*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>���]       a[��	��2*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss*�>


accuracy_1)\?��\�]       a[��	S3*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?�X4�]       a[��	D3*_��A�*O

prediction_loss���>

reg_loss|�w<


total_loss��>


accuracy_1�?�#�u]       a[��	�-3*_��A�*O

prediction_loss=
�>

reg_lossp�w<


total_loss���>


accuracy_1�z?�/X]       a[��	�=3*_��A�*O

prediction_loss���>

reg_lossg�w<


total_loss
��>


accuracy_1�?�!#�]       a[��	3O3*_��A�*O

prediction_loss�?

reg_loss`�w<


total_lossv�?


accuracy_1���>�)]       a[��	R�3*_��A�*O

prediction_lossq=
?

reg_lossK�w<


total_loss.?


accuracy_1��>"O�-]       a[��	��3*_��A�*O

prediction_loss��>

reg_loss?�w<


total_loss�D�>


accuracy_1q=
?j\Ʋ]       a[��	�3*_��A�*O

prediction_loss   ?

reg_loss6�w<


total_loss��?


accuracy_1   ??��]       a[��	��3*_��A�*O

prediction_loss���>

reg_loss-�w<


total_lossF��>


accuracy_1��?yC�h]       a[��	��3*_��A�*O

prediction_loss���>

reg_loss"�w<


total_loss��>


accuracy_1�?��]       a[��	�4*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�D�>


accuracy_1q=
?RW6�]       a[��	'4*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?C[pO]       a[��	�=4*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss-?


accuracy_1��>1�]       a[��	vT4*_��A�*O

prediction_loss{�>

reg_loss��w<


total_loss�ӵ>


accuracy_1��(?�]n�]       a[��	Lk4*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>Q(y]       a[��	�{4*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?g��z]       a[��	��4*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?.�]       a[��	*�4*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?8��K]       a[��	�4*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?G�9�]       a[��	��4*_��A�*O

prediction_loss{.?

reg_loss��w<


total_loss6�1?


accuracy_1
ף>y%{S]       a[��	��4*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossB��>


accuracy_1��?(u?�]       a[��	�	5*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?����]       a[��	�!5*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?Ꮘ�]       a[��	85*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?���]       a[��	jN5*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?�|��]       a[��	sd5*_��A�*O

prediction_loss���>

reg_loss|�w<


total_loss��>


accuracy_1�?�菍]       a[��	{�5*_��A�*O

prediction_loss���>

reg_lossq�w<


total_lossA��>


accuracy_1��?�hɔ]       a[��	w�5*_��A�*O

prediction_loss�?

reg_lossh�w<


total_lossr�?


accuracy_1���>��]       a[��	��5*_��A�*O

prediction_lossq=
?

reg_loss]�w<


total_loss*?


accuracy_1��>A�z�]       a[��	z�5*_��A�*O

prediction_loss�G�>

reg_lossR�w<


total_loss!�>


accuracy_1)\?����]       a[��	��5*_��A�*O

prediction_loss���>

reg_lossH�w<


total_loss��>


accuracy_1�?���]       a[��	�$6*_��A�*O

prediction_lossq=
?

reg_loss>�w<


total_loss*?


accuracy_1��>j��]       a[��	d@6*_��A�*O

prediction_loss\��>

reg_loss3�w<


total_loss�N�>


accuracy_1R�?c<>]       a[��	�f6*_��A�*O

prediction_loss��(?

reg_loss'�w<


total_loss|�,?


accuracy_1{�>��]       a[��	=�6*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss�>


accuracy_1)\?��h�]       a[��	ݱ6*_��A�*O

prediction_loss��(?

reg_loss�w<


total_loss{�,?


accuracy_1{�>̒�]       a[��	��6*_��A�*O

prediction_loss�?

reg_loss
�w<


total_lossp�?


accuracy_1���>bA#�]       a[��	��6*_��A�*O

prediction_loss��?

reg_loss��w<


total_lossRy?


accuracy_1���>�7 o]       a[��	e�6*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?\bZ2]       a[��	�7*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?��$�]       a[��	27*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss�>


accuracy_1)\?���S]       a[��	/N7*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss�>


accuracy_1)\?�Pm]       a[��	�f7*_��A�*O

prediction_loss�?

reg_loss��w<


total_losso�?


accuracy_1���>=<΢]       a[��	�}7*_��A�*O

prediction_loss�?

reg_loss��w<


total_losso�?


accuracy_1���>A�q�]       a[��	�7*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�D�>


accuracy_1q=
?SXdY]       a[��	2�7*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?�ւ�]       a[��	C�7*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>PL��]       a[��	/�7*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss:��>


accuracy_1��?�0z�]       a[��	��7*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss�>


accuracy_1)\?��ɫ]       a[��	8*_��A�*O

prediction_loss��(?

reg_loss�w<


total_lossy�,?


accuracy_1{�>�W��]       a[��	8*_��A�*O

prediction_loss��>

reg_lossv�w<


total_loss�D�>


accuracy_1q=
?$V�C]       a[��	�48*_��A�*O

prediction_loss=
�>

reg_lossk�w<


total_loss���>


accuracy_1�z?/���]       a[��	IK8*_��A�*O

prediction_lossq=
?

reg_lossa�w<


total_loss'?


accuracy_1��>1���]       a[��	�`8*_��A�*O

prediction_lossq=
?

reg_lossW�w<


total_loss&?


accuracy_1��>�rh�]       a[��	yw8*_��A�*O

prediction_loss   ?

reg_lossN�w<


total_loss��?


accuracy_1   ?J�J]       a[��	�8*_��A�*O

prediction_loss�Q�>

reg_lossC�w<


total_lossV�>


accuracy_1
�#?p��]       a[��	�8*_��A�*O

prediction_loss��>

reg_loss;�w<


total_loss�D�>


accuracy_1q=
?f+�q]       a[��	A�8*_��A�*O

prediction_loss���>

reg_loss/�w<


total_loss6��>


accuracy_1��?�JSP]       a[��	1�8*_��A�*O

prediction_loss)\?

reg_loss%�w<


total_loss�;?


accuracy_1�G�>�F��]       a[��	��8*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1�?�SV�]       a[��	z�8*_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Z?


accuracy_1=
�>m�I�]       a[��	{9*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss�>


accuracy_1)\?�È#]       a[��	^,9*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�2C�]       a[��	�E9*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?��!]       a[��	�[9*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_lossS�>


accuracy_1
�#?�<�]       a[��	3p9*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss$?


accuracy_1��>.l�]       a[��	��9*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss$?


accuracy_1��>pr�0]       a[��	��9*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossk�?


accuracy_1���>XDt]       a[��	��9*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossk�?


accuracy_1���>�]       a[��	}�9*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss$?


accuracy_1��>m�]       a[��	&�9*_��A�*O

prediction_loss��?

reg_loss��w<


total_lossMy?


accuracy_1���>ڨ�]       a[��	��9*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�f��]       a[��	�:*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?Srf�]       a[��	p(:*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�I]       a[��	�?:*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>u��]       a[��	 W:*_��A�*O

prediction_loss���>

reg_lossu�w<


total_loss1��>


accuracy_1��?C��`]       a[��	�l:*_��A�*O

prediction_loss��?

reg_lossj�w<


total_lossLy?


accuracy_1���>%2��]       a[��	��:*_��A�*O

prediction_loss���>

reg_loss`�w<


total_loss��>


accuracy_1�?�P��]       a[��	�:*_��A�*O

prediction_loss���>

reg_lossU�w<


total_loss��>


accuracy_1�?�$��]       a[��	�:*_��A�*O

prediction_loss���>

reg_lossK�w<


total_loss��>


accuracy_1�?%h@]       a[��	t�:*_��A�*O

prediction_loss��>

reg_lossA�w<


total_loss�D�>


accuracy_1q=
?�ԕ]       a[��	G�:*_��A�*O

prediction_loss��>

reg_loss7�w<


total_loss�D�>


accuracy_1q=
?X��]       a[��	;*_��A�*O

prediction_loss   ?

reg_loss-�w<


total_loss��?


accuracy_1   ?�g��]       a[��	�7;*_��A�*O

prediction_loss
ף>

reg_loss%�w<


total_lossk��>


accuracy_1{.?2ƙ]       a[��	rN;*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?��]       a[��	�f;*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1�?p�@U]       a[��	K�;*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?4m�]       a[��	?�;*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossh�?


accuracy_1���>��u]       a[��	��;*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?W.f]       a[��	z�;*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss!?


accuracy_1��>���]       a[��	��;*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss�>


accuracy_1)\?��7]       a[��	<*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss~D�>


accuracy_1q=
?^�A]       a[��	L3<*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�q��]       a[��	�J<*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?`,�]       a[��	Jb<*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss+��>


accuracy_1��?G�=]       a[��	F|<*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>;aˮ]       a[��	��<*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>��ȩ]       a[��	d�<*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss|D�>


accuracy_1q=
?�v]       a[��	��<*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>���R]       a[��	��<*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�Y+�]       a[��	��<*_��A�*O

prediction_loss=
�>

reg_losst�w<


total_loss���>


accuracy_1�z?Y�]       a[��	�=*_��A�*O

prediction_loss   ?

reg_lossk�w<


total_loss��?


accuracy_1   ?(�1�]       a[��	�'=*_��A�*O

prediction_loss�z?

reg_lossa�w<


total_loss�Z?


accuracy_1=
�>�^��]       a[��	�==*_��A�*O

prediction_loss�G�>

reg_lossV�w<


total_loss	�>


accuracy_1)\?a:8]       a[��	�U=*_��A�*O

prediction_loss   ?

reg_lossH�w<


total_loss��?


accuracy_1   ?Y��H]       a[��	v�=*_��A�*O

prediction_lossq=
?

reg_loss6�w<


total_loss?


accuracy_1��>+�3#]       a[��	f�=*_��A�*O

prediction_lossq=
?

reg_loss*�w<


total_loss?


accuracy_1��>C�]       a[��	�=*_��A�*O

prediction_loss�G�>

reg_loss �w<


total_loss�>


accuracy_1)\?٘v�]       a[��	�>*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?4H. ]       a[��	@4>*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss?


accuracy_1��>����]       a[��	L>*_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?K /]       a[��	�a>*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�]       a[��	�|>*_��A�*O

prediction_loss
ף>

reg_loss��w<


total_lossa��>


accuracy_1{.?���]       a[��	A�>*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>�.T�]       a[��	��>*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossc�?


accuracy_1���>��7]       a[��	��>*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>l_��]       a[��	�?*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>¨o]       a[��	|+?*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>���q]       a[��	FA?*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>�]       a[��	�W?*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>�0"]       a[��	�p?*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?��w�]       a[��	r�?*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>�'+�]       a[��	�?*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>wK!]       a[��	5�?*_��A�*O

prediction_loss�?

reg_lossz�w<


total_lossb�?


accuracy_1���>�o�p]       a[��	��?*_��A�*O

prediction_loss�G�>

reg_losso�w<


total_loss�>


accuracy_1)\?���]       a[��	� @*_��A�*O

prediction_loss
�#?

reg_lossd�w<


total_loss��'?


accuracy_1�Q�>@9�]       a[��	aQ@*_��A�*O

prediction_loss��>

reg_loss[�w<


total_lossrD�>


accuracy_1q=
?����]       a[��	��@*_��A�*O

prediction_loss��>

reg_lossO�w<


total_lossqD�>


accuracy_1q=
?g��]       a[��	�@*_��A�*O

prediction_loss)\?

reg_lossE�w<


total_loss�;?


accuracy_1�G�>DY��]       a[��	��@*_��A�*O

prediction_loss��>

reg_loss:�w<


total_lossqD�>


accuracy_1q=
?����]       a[��	��@*_��A�*O

prediction_loss   ?

reg_loss0�w<


total_loss��?


accuracy_1   ?H�]       a[��	)A*_��A�*O

prediction_loss�?

reg_loss&�w<


total_lossa�?


accuracy_1���>����]       a[��	�"A*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss?


accuracy_1��>$��]       a[��	�<A*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1��?�qD�]       a[��	�WA*_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�;?


accuracy_1�G�>��0T]       a[��	�sA*_��A�*O

prediction_loss�Q8?

reg_loss��w<


total_loss�1<?


accuracy_1)\�>#?+�]       a[��	7�A*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>z�TQ]       a[��	�A*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>/9�]       a[��	A�A*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?a��]       a[��	��A*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>�+�]       a[��	��A*_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�N�>


accuracy_1R�?,z]       a[��	a�A*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>�	��]       a[��	aB*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>��O�]       a[��	/B*_��A�*O

prediction_loss��>

reg_loss��w<


total_losslD�>


accuracy_1q=
?,d��]       a[��	�GB*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss_�?


accuracy_1���>��P�]       a[��	�cB*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss܁�>


accuracy_1�?�O]       a[��	|B*_��A�*O

prediction_loss��>

reg_loss��w<


total_losskD�>


accuracy_1q=
?��;]       a[��	��B*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss^�?


accuracy_1���>�NWf]       a[��	��B*_��A�*O

prediction_loss�?

reg_lossy�w<


total_loss^�?


accuracy_1���>f��]       a[��	v�B*_��A�*O

prediction_loss�G�>

reg_lossm�w<


total_loss��>


accuracy_1)\?��G�]       a[��	<�B*_��A�*O

prediction_lossq=
?

reg_lossc�w<


total_loss?


accuracy_1��>Nx�]       a[��	'�B*_��A�*O

prediction_loss�G�>

reg_lossX�w<


total_loss��>


accuracy_1)\?Ζ�]       a[��	C*_��A�*O

prediction_loss   ?

reg_lossN�w<


total_loss��?


accuracy_1   ?�Sj�]       a[��	�+C*_��A�*O

prediction_loss   ?

reg_lossB�w<


total_loss��?


accuracy_1   ?<��]       a[��	BC*_��A�*O

prediction_loss)\?

reg_loss9�w<


total_loss�;?


accuracy_1�G�>�n]       a[��	�XC*_��A�*O

prediction_loss��>

reg_loss0�w<


total_losshD�>


accuracy_1q=
?O�y�]       a[��	�wC*_��A�*O

prediction_loss�G�>

reg_loss%�w<


total_loss��>


accuracy_1)\?p���]       a[��	��C*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss?


accuracy_1��>�]��]       a[��	�C*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss\�?


accuracy_1���>W=�~]       a[��	��C*_��A�*O

prediction_loss���>

reg_loss�w<


total_lossׁ�>


accuracy_1�?�镯]       a[��	��C*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss\�?


accuracy_1���>���H]       a[��	�D*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>~���]       a[��	�D*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossց�>


accuracy_1�?�!��]       a[��	r5D*_��A�*O

prediction_loss��(?

reg_loss��w<


total_lossg�,?


accuracy_1{�>J5�9]       a[��	�OD*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>v/��]       a[��	�gD*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss=y?


accuracy_1���>{��S]       a[��	I�D*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss=y?


accuracy_1���>�Ѵ�]       a[��	$�D*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�{��]       a[��	}�D*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss=y?


accuracy_1���>�l�,]       a[��	\�D*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>��ڕ]       a[��	��D*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Z?


accuracy_1=
�>*֑�]       a[��	#�D*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?x�Y]       a[��	E*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossZ�?


accuracy_1���>Ʉ��]       a[��	�$E*_��A�*O

prediction_loss��>

reg_lossw�w<


total_losscD�>


accuracy_1q=
?���]       a[��	\;E*_��A�*O

prediction_loss���>

reg_lossm�w<


total_loss��>


accuracy_1��?l�"�]       a[��	�TE*_��A�*O

prediction_loss   ?

reg_lossd�w<


total_loss��?


accuracy_1   ?��]       a[��	lE*_��A�*O

prediction_loss��?

reg_lossW�w<


total_loss;y?


accuracy_1���>�v]       a[��	M�E*_��A�*O

prediction_loss���>

reg_lossN�w<


total_loss��>


accuracy_1��?q��r]       a[��	x�E*_��A�*O

prediction_lossq=
?

reg_lossC�w<


total_loss?


accuracy_1��>h,q�]       a[��	��E*_��A�*O

prediction_loss�z?

reg_loss;�w<


total_loss�Z?


accuracy_1=
�>u�4�]       a[��	��E*_��A�*O

prediction_loss
�#?

reg_loss-�w<


total_loss��'?


accuracy_1�Q�>��2Y]       a[��	��E*_��A�*O

prediction_loss��>

reg_loss#�w<


total_loss`D�>


accuracy_1q=
?V�p]       a[��	��E*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss?


accuracy_1��>�"V�]       a[��	F*_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Z?


accuracy_1=
�>��e]       a[��	�-F*_��A�*O

prediction_loss333?

reg_loss�w<


total_loss�7?


accuracy_1���>��]       a[��	�GF*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossρ�>


accuracy_1�?���]       a[��	�`F*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>�@q]       a[��	�yF*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?N��A]       a[��	��F*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?\�R]       a[��	ǹF*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��> ���]       a[��	��F*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss́�>


accuracy_1�?s<_�]       a[��	��F*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?T��]       a[��	� G*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?U�]       a[��	�G*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss\D�>


accuracy_1q=
?��]       a[��	/G*_��A�*O

prediction_loss��(?

reg_loss��w<


total_lossa�,?


accuracy_1{�> QN�]       a[��	8IG*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>���]       a[��	�`G*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�6�]       a[��	�{G*_��A�*O

prediction_loss=
�>

reg_loss|�w<


total_lossy��>


accuracy_1�z?�p:]       a[��	őG*_��A�*O

prediction_loss   ?

reg_losss�w<


total_loss��?


accuracy_1   ?�k$ ]       a[��	*�G*_��A�*O

prediction_loss�z?

reg_lossh�w<


total_lossZ?


accuracy_1=
�>�u�l]       a[��	ƿG*_��A�*O

prediction_loss�G�>

reg_loss]�w<


total_loss��>


accuracy_1)\?g� T]       a[��	��G*_��A�*O

prediction_lossq=
?

reg_lossT�w<


total_loss?


accuracy_1��>[&,]       a[��	��G*_��A�*O

prediction_loss=
�>

reg_lossK�w<


total_lossw��>


accuracy_1�z?��d�]       a[��	JH*_��A�*O

prediction_lossq=
?

reg_loss@�w<


total_loss?


accuracy_1��>��
]       a[��	�!H*_��A�*O

prediction_loss���>

reg_loss5�w<


total_lossɁ�>


accuracy_1�?�f�=]       a[��	itH*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss?


accuracy_1��>؁ݵ]       a[��	��H*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss?


accuracy_1��>ܠϚ]       a[��	��H*_��A�*O

prediction_loss)\?

reg_loss
�w<


total_loss�;?


accuracy_1�G�>'�}�]       a[��	�H*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossWD�>


accuracy_1q=
?j�]       a[��	��H*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?T�=�]       a[��	��H*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss}Z?


accuracy_1=
�>BP9]       a[��	CI*_��A�*O

prediction_loss)\�>

reg_loss��w<


total_loss`�>


accuracy_1�Q8?���]       a[��	fI*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�j�]       a[��	�/I*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1��?J�A]       a[��	�FI*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss"�>


accuracy_1
�#?�i��]       a[��	bI*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossS�?


accuracy_1���>󑎥]       a[��	�zI*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>ƻ�W]       a[��	͔I*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss|Z?


accuracy_1=
�>���]       a[��	ɬI*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>��]       a[��		�I*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>�Nx�]       a[��	b�I*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossR�?


accuracy_1���>V���]       a[��	0�I*_��A�*O

prediction_lossq=
?

reg_loss{�w<


total_loss?


accuracy_1��>�~0]       a[��	�J*_��A�*O

prediction_loss)\?

reg_lossp�w<


total_loss�;?


accuracy_1�G�>�!w]       a[��	�(J*_��A�*O

prediction_loss�?

reg_lossg�w<


total_lossR�?


accuracy_1���>Q8�]       a[��	cDJ*_��A�*O

prediction_loss���>

reg_loss\�w<


total_loss�>


accuracy_1�?�z=Y]       a[��	qJ*_��A�*O

prediction_loss���>

reg_lossR�w<


total_loss�>


accuracy_1�?��]s]       a[��	��J*_��A�*O

prediction_loss���>

reg_lossF�w<


total_loss���>


accuracy_1�?=BP�]       a[��	)�J*_��A�*O

prediction_loss��>

reg_loss=�w<


total_lossQD�>


accuracy_1q=
?Z!��]       a[��	K*_��A�*O

prediction_loss   ?

reg_loss2�w<


total_loss��?


accuracy_1   ?ӜP�]       a[��	�BK*_��A�*O

prediction_loss   ?

reg_loss(�w<


total_loss��?


accuracy_1   ?Ѱ8]       a[��	R�K*_��A�*O

prediction_loss��>

reg_loss�w<


total_lossPD�>


accuracy_1q=
?vl�k]       a[��	��K*_��A�*O

prediction_loss�?

reg_loss�w<


total_lossP�?


accuracy_1���>_�Uo]       a[��	^.L*_��A�*O

prediction_loss�p=?

reg_loss
�w<


total_loss<PA?


accuracy_1��>אrJ]       a[��	vL*_��A�*O

prediction_loss)\?

reg_loss �w<


total_loss�;?


accuracy_1�G�>ǥx�]       a[��	m�L*_��A�*O

prediction_loss
ף>

reg_loss��w<


total_loss:��>


accuracy_1{.?2H]       a[��	��L*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss	?


accuracy_1��>�h�)]       a[��	XM*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?`�]       a[��	UM*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossl��>


accuracy_1�z?4WL�]       a[��	��M*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossk��>


accuracy_1�z?�'�H]       a[��	��M*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?���]       a[��	?�M*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?6�]       a[��	�N*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossO�?


accuracy_1���>T�]       a[��	:WN*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossj��>


accuracy_1�z?h�\]       a[��	{�N*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossLD�>


accuracy_1q=
?*��]       a[��	`�N*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss0y?


accuracy_1���>�A62]       a[��	>	O*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossi��>


accuracy_1�z?�a��]       a[��	�BO*_��A�*O

prediction_lossq=
?

reg_loss|�w<


total_loss?


accuracy_1��>��y]       a[��	hO*_��A�*O

prediction_lossq=
?

reg_lossq�w<


total_loss?


accuracy_1��>�^�]       a[��	��O*_��A�*O

prediction_loss   ?

reg_lossf�w<


total_loss��?


accuracy_1   ?60�]       a[��	�O*_��A�*O

prediction_loss   ?

reg_loss\�w<


total_loss��?


accuracy_1   ?x���]       a[��	��O*_��A�*O

prediction_lossq=
?

reg_lossP�w<


total_loss?


accuracy_1��>�	�]       a[��	L7P*_��A�*O

prediction_loss=
�>

reg_lossG�w<


total_lossg��>


accuracy_1�z?�N��]       a[��	�_P*_��A�*O

prediction_loss�?

reg_loss;�w<


total_lossM�?


accuracy_1���>�LY�]       a[��	�{P*_��A�*O

prediction_loss��>

reg_loss/�w<


total_lossHD�>


accuracy_1q=
?+1�(]       a[��	V�P*_��A�*O

prediction_loss�?

reg_loss(�w<


total_lossM�?


accuracy_1���>ݧO.]       a[��	��P*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?���:]       a[��	�Q*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss?


accuracy_1��>�C�"]       a[��	1AQ*_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�;?


accuracy_1�G�>�)fw]       a[��	,eQ*_��A�*O

prediction_loss
�#?

reg_loss��w<


total_loss��'?


accuracy_1�Q�> )��]       a[��	r�Q*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>�Fz]       a[��	:�Q*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossFD�>


accuracy_1q=
?�Vk�]       a[��	��Q*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?�/��]       a[��	�R*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>�6�Q]       a[��	�QR*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>>b�]       a[��	��R*_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss�"?


accuracy_1\��>H���]       a[��	6�R*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossED�>


accuracy_1q=
?v��]       a[��	Z�R*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossK�?


accuracy_1���>�*�]       a[��	4�R*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossK�?


accuracy_1���>��o�]       a[��	8S*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?����]       a[��	�QS*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossa��>


accuracy_1�z?�p�N]       a[��	M�S*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?��V�]       a[��	�S*_��A�*O

prediction_loss\��>

reg_lossy�w<


total_loss�N�>


accuracy_1R�?37��]       a[��	��S*_��A�*O

prediction_loss���>

reg_losso�w<


total_loss���>


accuracy_1�?x]       a[��	h	T*_��A�*O

prediction_loss���>

reg_lossd�w<


total_loss���>


accuracy_1�?���]       a[��	#/T*_��A�*O

prediction_lossq=
?

reg_lossZ�w<


total_loss?


accuracy_1��>ޔ]       a[��	�VT*_��A�*O

prediction_loss�G�>

reg_lossO�w<


total_loss��>


accuracy_1)\?��.]       a[��	�T*_��A�*O

prediction_loss)\?

reg_lossE�w<


total_loss�;?


accuracy_1�G�>)�4�]       a[��	��T*_��A�*O

prediction_loss���>

reg_loss8�w<


total_loss���>


accuracy_1�?G��]       a[��	�T*_��A�*O

prediction_loss   ?

reg_loss/�w<


total_loss��?


accuracy_1   ?���E]       a[��	Y�T*_��A�*O

prediction_loss�G�>

reg_loss%�w<


total_loss��>


accuracy_1)\?G�V�]       a[��	��T*_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�;?


accuracy_1�G�>���]       a[��	�U*_��A�*O

prediction_loss
�#?

reg_loss�w<


total_loss��'?


accuracy_1�Q�>��4]       a[��	DU*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?}��J]       a[��	�gU*_��A�*O

prediction_loss{.?

reg_loss��w<


total_loss�1?


accuracy_1
ף>4��]       a[��	ÂU*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?S#�]       a[��	��U*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss>D�>


accuracy_1q=
?���D]       a[��	��U*_��A�*O

prediction_loss�?

reg_loss��w<


total_lossG�?


accuracy_1���>� ?]       a[��	0�U*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?�wY]       a[��	h�U*_��A�*O

prediction_loss\��>

reg_loss��w<


total_losszN�>


accuracy_1R�?�l�]]       a[��	dV*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss)y?


accuracy_1���>���]       a[��	V*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>��[�]       a[��	�6V*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossZ��>


accuracy_1�z?���]       a[��	?SV*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>F�e�]       a[��	'lV*_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossoZ?


accuracy_1=
�>P1��]       a[��	4�V*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?ù�]       a[��	�V*_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossoZ?


accuracy_1=
�>k���]       a[��	(�V*_��A�*O

prediction_loss��(?

reg_losss�w<


total_lossQ�,?


accuracy_1{�>�}.�]       a[��	��V*_��A�*O

prediction_loss�z?

reg_lossj�w<


total_lossoZ?


accuracy_1=
�>x�G�]       a[��	��V*_��A�*O

prediction_loss���>

reg_loss_�w<


total_loss��>


accuracy_1��?�a-�]       a[��	�W*_��A�*O

prediction_loss��>

reg_lossS�w<


total_loss:D�>


accuracy_1q=
?�VX�]       a[��	�W*_��A�*O

prediction_loss�G�>

reg_lossI�w<


total_loss��>


accuracy_1)\?t]       a[��	{3W*_��A�*O

prediction_loss�?

reg_loss@�w<


total_lossE�?


accuracy_1���>�^�~]       a[��	KW*_��A�*O

prediction_loss���>

reg_loss6�w<


total_loss���>


accuracy_1�?�ᱟ]       a[��	�aW*_��A�*O

prediction_loss=
�>

reg_loss*�w<


total_lossV��>


accuracy_1�z?.��]       a[��	{W*_��A�*O

prediction_loss{�>

reg_loss!�w<


total_loss�ӵ>


accuracy_1��(?>��]       a[��	��W*_��A�*O

prediction_loss��?

reg_loss�w<


total_loss&y?


accuracy_1���>��K]       a[��	��W*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?uh[v]       a[��	 �W*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?��W.]       a[��	S	X*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?CT� ]       a[��	�X*_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossޗ"?


accuracy_1\��>��!�]       a[��	�7X*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss%y?


accuracy_1���>"#	�]       a[��	vOX*_��A�*O

prediction_loss�z?

reg_loss��w<


total_losslZ?


accuracy_1=
�>/�.]       a[��	MjX*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss5D�>


accuracy_1q=
?�pK�]       a[��	��X*_��A�*O

prediction_loss
�#?

reg_loss��w<


total_loss��'?


accuracy_1�Q�>�?�0]       a[��	�X*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss5D�>


accuracy_1q=
?��U]       a[��	��X*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?-c�;]       a[��	��X*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?s|g]       a[��	�X*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1��?#Q�]       a[��	�X*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss3D�>


accuracy_1q=
?�W��]       a[��	Y*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1��?���a]       a[��	%Y*_��A�*O

prediction_loss�z?

reg_lossw�w<


total_losskZ?


accuracy_1=
�>v���]       a[��	 DY*_��A�*O

prediction_loss���>

reg_lossl�w<


total_loss���>


accuracy_1�?�'�]       a[��	1\Y*_��A�*O

prediction_loss�G�>

reg_lossc�w<


total_loss��>


accuracy_1)\?��]       a[��	�zY*_��A�*O

prediction_loss)\?

reg_lossX�w<


total_loss�;?


accuracy_1�G�>肝]       a[��	!�Y*_��A�*O

prediction_loss)\?

reg_lossN�w<


total_loss�;?


accuracy_1�G�>����]       a[��	K�Y*_��A�*O

prediction_loss=
�>

reg_lossC�w<


total_lossO��>


accuracy_1�z?���]       a[��	B�Y*_��A�*O

prediction_loss�G�>

reg_loss9�w<


total_loss��>


accuracy_1)\?��]       a[��	M�Y*_��A�*O

prediction_loss)\?

reg_loss/�w<


total_loss�;?


accuracy_1�G�>w���]       a[��	�Z*_��A�*O

prediction_loss�G�>

reg_loss#�w<


total_loss��>


accuracy_1)\?fyH�]       a[��	�?Z*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss@�?


accuracy_1���>�5��]       a[��	+hZ*_��A�*O

prediction_loss
�#?

reg_loss�w<


total_loss��'?


accuracy_1�Q�>#�QS]       a[��	t�Z*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss/D�>


accuracy_1q=
?��1k]       a[��	��Z*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>>t�]       a[��	��Z*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossL��>


accuracy_1�z?Z6j�]       a[��	y[*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?��d]       a[��	�[*_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossٗ"?


accuracy_1\��>��[]       a[��	�6[*_��A�*O

prediction_loss�z?

reg_loss��w<


total_losshZ?


accuracy_1=
�>,S7e]       a[��	�T[*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?lw\
]       a[��	�[*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?��B]       a[��	p�[*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?/�']       a[��	��[*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>�@��]       a[��	B\*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossڋ�>


accuracy_1��?7J�L]       a[��	�+\*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?r�]       a[��	�K\*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss+D�>


accuracy_1q=
?��]       a[��	�d\*_��A�*O

prediction_loss
�#?

reg_loss~�w<


total_loss��'?


accuracy_1�Q�>��{]       a[��	�\*_��A�*O

prediction_loss=
�>

reg_losss�w<


total_lossI��>


accuracy_1�z?��v]       a[��	��\*_��A�*O

prediction_lossR�?

reg_lossj�w<


total_lossؗ"?


accuracy_1\��>�t�"]       a[��	c�\*_��A�*O

prediction_loss�G�>

reg_loss_�w<


total_loss��>


accuracy_1)\?"���]       a[��	��\*_��A�*O

prediction_lossq=
?

reg_lossV�w<


total_loss�?


accuracy_1��>�h
�]       a[��	5�\*_��A�*O

prediction_loss�?

reg_lossK�w<


total_loss=�?


accuracy_1���>��c�]       a[��	]*_��A�*O

prediction_loss�z?

reg_lossA�w<


total_lossfZ?


accuracy_1=
�>����]       a[��	T]*_��A�*O

prediction_loss�G�>

reg_loss6�w<


total_loss��>


accuracy_1)\?	)G�]       a[��	�7]*_��A�*O

prediction_loss��>

reg_loss-�w<


total_loss(D�>


accuracy_1q=
?�e;�]       a[��	�P]*_��A�*O

prediction_lossq=
?

reg_loss"�w<


total_loss�?


accuracy_1��>kmci]       a[��	Yk]*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss<�?


accuracy_1���>|pU/]       a[��	��]*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1�?wg�j]       a[��	s�]*_��A�*O

prediction_loss��?

reg_loss�w<


total_lossy?


accuracy_1���>�" _]       a[��	�]*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?<��t]       a[��	��]*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?�u(P]       a[��	�^*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>��]       a[��	� ^*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossԋ�>


accuracy_1��?�q�]       a[��	2:^*_��A�*O

prediction_loss\��>

reg_loss��w<


total_lossbN�>


accuracy_1R�?1���]       a[��	�R^*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossӋ�>


accuracy_1��?"���]       a[��	�i^*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?���]       a[��	#�^*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?�r<�]       a[��	�^*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>8��n]       a[��	��^*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?����]       a[��	T�^*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?����]       a[��	*�^*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?���]       a[��	�_*_��A�*O

prediction_loss�z?

reg_lossz�w<


total_losscZ?


accuracy_1=
�>z�]       a[��	Q1_*_��A�*O

prediction_loss��?

reg_lossp�w<


total_lossy?


accuracy_1���>}pԭ]       a[��	�J_*_��A�*O

prediction_loss\��>

reg_losse�w<


total_loss_N�>


accuracy_1R�?��Y]       a[��	a_*_��A�*O

prediction_loss���>

reg_lossZ�w<


total_loss���>


accuracy_1�?�^T�]       a[��	�v_*_��A�*O

prediction_loss���>

reg_lossO�w<


total_loss���>


accuracy_1�?����]       a[��	z�_*_��A�*O

prediction_loss�G�>

reg_lossF�w<


total_loss��>


accuracy_1)\?��� ]       a[��	ʧ_*_��A�*O

prediction_loss
�#?

reg_loss<�w<


total_loss��'?


accuracy_1�Q�>��.]       a[��	L�_*_��A�*O

prediction_loss��>

reg_loss1�w<


total_loss!D�>


accuracy_1q=
?W��]       a[��	'�_*_��A�*O

prediction_loss�?

reg_loss(�w<


total_loss9�?


accuracy_1���>�_(]       a[��	��_*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1�?,Vg�]       a[��	`*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?��l0]       a[��	�%`*_��A�*O

prediction_loss��(?

reg_loss�w<


total_lossC�,?


accuracy_1{�>��@b]       a[��	B>`*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?�G�S]       a[��	�Y`*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossD�>


accuracy_1q=
?a���]       a[��	�s`*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?s�=�]       a[��	G�`*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?}��]       a[��	�`*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?� �)]       a[��	��`*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?����]       a[��	�`*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>��b]       a[��	)a*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?��y�]       a[��	7Oa*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>���)]       a[��	�qa*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?��r�]       a[��	`�a*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss~�?


accuracy_1   ?���w]       a[��	��a*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>�cR]       a[��	N�a*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss~�?


accuracy_1   ?6>��]       a[��	?b*_��A�*O

prediction_loss���>

reg_lossv�w<


total_lossɋ�>


accuracy_1��?�(�]       a[��	�Cb*_��A�*O

prediction_loss���>

reg_lossl�w<


total_loss���>


accuracy_1�?����]       a[��	�db*_��A�*O

prediction_loss��>

reg_lossb�w<


total_lossD�>


accuracy_1q=
?$�b�]       a[��	ѓb*_��A�*O

prediction_loss���>

reg_lossX�w<


total_loss���>


accuracy_1�?�Rz1]       a[��	��b*_��A�*O

prediction_loss�G�>

reg_lossN�w<


total_loss��>


accuracy_1)\?^`�&]       a[��	��b*_��A�*O

prediction_loss�?

reg_lossE�w<


total_loss5�?


accuracy_1���>�Ü�]       a[��	0c*_��A�*O

prediction_lossq=
?

reg_loss9�w<


total_loss�?


accuracy_1��>F��4]       a[��	?c*_��A�*O

prediction_lossq=
?

reg_loss/�w<


total_loss�?


accuracy_1��>>v<�]       a[��	bjc*_��A�*O

prediction_lossq=
?

reg_loss$�w<


total_loss�?


accuracy_1��>[Z�~]       a[��	ӥc*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?��a*]       a[��	��c*_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss5��>


accuracy_1�z?�#�q]       a[��	]pd*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>6�)]       a[��	ݔd*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?m�%�]       a[��	��d*_��A�*O

prediction_loss��?

reg_loss��w<


total_lossy?


accuracy_1���>I��]       a[��	��d*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossD�>


accuracy_1q=
?E�{�]       a[��	��d*_��A�*O

prediction_loss��?

reg_loss��w<


total_lossy?


accuracy_1���>ů7 ]       a[��	$e*_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss͗"?


accuracy_1\��>+Hy]       a[��	j�e*_��A�*O

prediction_loss333?

reg_loss��w<


total_loss�7?


accuracy_1���>�8�]       a[��	��e*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>����]       a[��	��e*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?��H�]       a[��	�e*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�>


accuracy_1��?��J]       a[��	*f*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>e��o]       a[��	��f*_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossz�?


accuracy_1   ?���c]       a[��	g*_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss̗"?


accuracy_1\��>B^{]       a[��	�6g*_��A�*O

prediction_loss)\?

reg_lossv�w<


total_loss�;?


accuracy_1�G�>g���]       a[��	��g*_��A�*O

prediction_loss���>

reg_lossm�w<


total_loss���>


accuracy_1��?hQ��]       a[��	�h*_��A�*O

prediction_loss   ?

reg_lossa�w<


total_lossz�?


accuracy_1   ?9�ͮ]       a[��	�Ah*_��A�*O

prediction_lossq=
?

reg_lossX�w<


total_loss�?


accuracy_1��>��b]       a[��	sih*_��A�*O

prediction_loss���>

reg_lossN�w<


total_loss���>


accuracy_1�? �-0]       a[��	t�h*_��A�*O

prediction_loss�G�>

reg_lossC�w<


total_loss��>


accuracy_1)\?�3��]       a[��	L�h*_��A�*O

prediction_lossq=
?

reg_loss7�w<


total_loss�?


accuracy_1��>C��p]       a[��	�h*_��A�*O

prediction_loss�?

reg_loss-�w<


total_loss1�?


accuracy_1���>8��]       a[��	28i*_��A�*O

prediction_loss�?

reg_loss$�w<


total_loss1�?


accuracy_1���>�t��]       a[��	/ji*_��A�*O

prediction_lossR�?

reg_loss�w<


total_lossʗ"?


accuracy_1\��>���]       a[��	�i*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>H���]       a[��	��i*_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�;?


accuracy_1�G�>Q��]       a[��	�i*_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossYZ?


accuracy_1=
�>�08]       a[��	�"j*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>�]       a[��	<Lj*_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossYZ?


accuracy_1=
�>B�=M]       a[��	Ynj*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>����]       a[��	��j*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?j&b]       a[��	*�j*_��A�*O

prediction_loss��?

reg_loss��w<


total_lossy?


accuracy_1���>f��:]       a[��	��j*_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossɗ"?


accuracy_1\��>)`�]       a[��	,k*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossD�>


accuracy_1q=
?�T]       a[��	<Kk*_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossw�?


accuracy_1   ?�]       a[��	�tk*_��A�*O

prediction_loss��?

reg_loss��w<


total_lossy?


accuracy_1���>t��]       a[��	_�k*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?�D$|]       a[��	��k*_��A�*O

prediction_loss{.?

reg_loss��w<


total_loss��1?


accuracy_1
ף>#�1�]       a[��	D�k*_��A�*O

prediction_loss   ?

reg_loss}�w<


total_lossv�?


accuracy_1   ?r`�]       a[��	�Wl*_��A�*O

prediction_loss���>

reg_losss�w<


total_loss{��>


accuracy_1�?��B]       a[��	I�l*_��A�*O

prediction_loss���>

reg_lossi�w<


total_lossz��>


accuracy_1�?��&]       a[��	�l*_��A�*O

prediction_loss���>

reg_loss]�w<


total_lossz��>


accuracy_1�?EQ��]       a[��	��l*_��A�*O

prediction_loss���>

reg_lossS�w<


total_loss���>


accuracy_1��?��]       a[��	qm*_��A�*O

prediction_loss���>

reg_lossI�w<


total_lossy��>


accuracy_1�?�6�]       a[��	�Cm*_��A�*O

prediction_loss���>

reg_loss@�w<


total_lossy��>


accuracy_1�?P2]       a[��	em*_��A�*O

prediction_lossR�?

reg_loss5�w<


total_lossǗ"?


accuracy_1\��>��	N]       a[��	:�m*_��A�*O

prediction_loss���>

reg_loss+�w<


total_lossx��>


accuracy_1�?.�%�]       a[��	��m*_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss&��>


accuracy_1�z?9{q]       a[��	��m*_��A�*O

prediction_loss   ?

reg_loss�w<


total_losst�?


accuracy_1   ?
�]       a[��	n*_��A�*O

prediction_loss�z?

reg_loss�w<


total_lossUZ?


accuracy_1=
�>�{A�]       a[��	N(n*_��A�*O

prediction_loss   ?

reg_loss�w<


total_losst�?


accuracy_1   ?r�|�]       a[��	�Yn*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossw��>


accuracy_1�?g�x�]       a[��	��n*_��A�	*O

prediction_loss���>

reg_loss��w<


total_lossv��>


accuracy_1�?hK]       a[��	Y�n*_��A�	*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?Ә�]       a[��	��n*_��A�	*O

prediction_lossR�?

reg_loss��w<


total_lossŗ"?


accuracy_1\��>*��A]       a[��	�o*_��A�	*O

prediction_loss=
�>

reg_loss��w<


total_loss#��>


accuracy_1�z?����]       a[��	�(o*_��A�	*O

prediction_loss=
�>

reg_loss��w<


total_loss#��>


accuracy_1�z?�<�]       a[��	Do*_��A�	*O

prediction_loss   ?

reg_loss��w<


total_losss�?


accuracy_1   ?_P�4]       a[��	/no*_��A�	*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>�#Y4]       a[��	�o*_��A�	*O

prediction_loss   ?

reg_loss��w<


total_losss�?


accuracy_1   ?����]       a[��	��o*_��A�	*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>K��]       a[��	��o*_��A�	*O

prediction_loss��?

reg_loss��w<


total_lossy?


accuracy_1���>����]       a[��	p*_��A�	*O

prediction_loss   ?

reg_loss��w<


total_lossr�?


accuracy_1   ?:��]       a[��	�.p*_��A�	*O

prediction_loss   ?

reg_loss|�w<


total_lossr�?


accuracy_1   ?��	 ]       a[��	�Np*_��A�	*O

prediction_loss   ?

reg_lossr�w<


total_lossr�?


accuracy_1   ?]�0�]       a[��	�zp*_��A�	*O

prediction_lossq=
?

reg_lossh�w<


total_loss�?


accuracy_1��>U#�]       a[��	J�p*_��A�	*O

prediction_loss�?

reg_loss]�w<


total_loss)�?


accuracy_1���>�{�Q]       a[��	��p*_��A�	*O

prediction_loss��>

reg_lossT�w<


total_lossD�>


accuracy_1q=
?
���]       a[��	��p*_��A�	*O

prediction_loss�z?

reg_lossH�w<


total_lossRZ?


accuracy_1=
�>���G]       a[��	#�p*_��A�	*O

prediction_loss��>

reg_loss<�w<


total_lossD�>


accuracy_1q=
?cs�b]       a[��	�q*_��A�	*O

prediction_loss�?

reg_loss2�w<


total_loss)�?


accuracy_1���>��`�]       a[��	s/q*_��A�	*O

prediction_loss���>

reg_loss)�w<


total_lossp��>


accuracy_1�?Wz�z]       a[��	�Hq*_��A�	*O

prediction_loss��>

reg_loss�w<


total_loss D�>


accuracy_1q=
?(~��]       a[��	�aq*_��A�	*O

prediction_loss)\?

reg_loss�w<


total_loss�;?


accuracy_1�G�>��_�]       a[��	_{q*_��A�	*O

prediction_loss   ?

reg_loss	�w<


total_lossp�?


accuracy_1   ?���]       a[��	h�q*_��A�	*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?�̪e]       a[��	��q*_��A�	*O

prediction_loss   ?

reg_loss��w<


total_lossp�?


accuracy_1   ??���]       a[��	��q*_��A�	*O

prediction_loss   ?

reg_loss��w<


total_lossp�?


accuracy_1   ?{".?]       a[��	n�q*_��A�	*O

prediction_loss)\�>

reg_loss��w<


total_loss�>


accuracy_1�Q8?��^�]       a[��	#�q*_��A�	*O

prediction_loss��?

reg_loss��w<


total_loss	y?


accuracy_1���>�s�]       a[��	�r*_��A�	*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?�?�]       a[��	�-r*_��A�	*O

prediction_loss���>

reg_loss��w<


total_lossm��>


accuracy_1�?]7]       a[��	�Er*_��A�	*O

prediction_loss���>

reg_loss��w<


total_lossm��>


accuracy_1�?<��]       a[��	 _r*_��A�	*O

prediction_loss   ?

reg_loss��w<


total_losso�?


accuracy_1   ?cb��]       a[��	�zr*_��A�	*O

prediction_loss   ?

reg_loss��w<


total_losso�?


accuracy_1   ?�+p�]       a[��	�r*_��A�	*O

prediction_loss��?

reg_loss��w<


total_lossy?


accuracy_1���>�Z]       a[��	"�r*_��A�	*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>m��]       a[��	T�r*_��A�	*O

prediction_loss��?

reg_loss��w<


total_lossy?


accuracy_1���>A�]       a[��	8�r*_��A�	*O

prediction_loss��>

reg_lossx�w<


total_loss�C�>


accuracy_1q=
?+-X�]       a[��	|�r*_��A�	*O

prediction_lossq=
?

reg_lossn�w<


total_loss�?


accuracy_1��>g���]       a[��	)s*_��A�	*O

prediction_loss��?

reg_lossc�w<


total_lossy?


accuracy_1���>9K�/]       a[��	�s*_��A�	*O

prediction_loss�G�>

reg_lossZ�w<


total_loss��>


accuracy_1)\?���]       a[��	!;s*_��A�	*O

prediction_loss��>

reg_lossM�w<


total_loss�C�>


accuracy_1q=
?��ց]       a[��	�Ns*_��A�	*O

prediction_loss�?

reg_lossD�w<


total_loss%�?


accuracy_1���>Mǲ�]       a[��	�cs*_��A�	*O

prediction_loss\��>

reg_loss:�w<


total_loss6N�>


accuracy_1R�?�g�E]       a[��	�ys*_��A�	*O

prediction_loss�G�>

reg_loss/�w<


total_loss��>


accuracy_1)\?o~�q]       a[��	�s*_��A�	*O

prediction_loss�Q�>

reg_loss"�w<


total_loss��>


accuracy_1
�#?`�_�]       a[��	r�s*_��A�	*O

prediction_loss��?

reg_loss�w<


total_lossy?


accuracy_1���>F��S]       a[��	��s*_��A�	*O

prediction_loss�Q�>

reg_loss�w<


total_loss��>


accuracy_1
�#?�cQ�]       a[��	��s*_��A�	*O

prediction_loss���>

reg_loss�w<


total_lossg��>


accuracy_1�?Oh��]       a[��	��s*_��A�	*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>,���]       a[��	?t*_��A�	*O

prediction_loss\��>

reg_loss��w<


total_loss3N�>


accuracy_1R�?�,]       a[��	6Vt*_��A�	*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?{�,�]       a[��	�nt*_��A�	*O

prediction_loss�?

reg_loss��w<


total_loss#�?


accuracy_1���>�6��]       a[��	
�t*_��A�	*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>�ą�]       a[��	�t*_��A�	*O

prediction_loss�?

reg_loss��w<


total_loss#�?


accuracy_1���>�~�5]       a[��	ɮt*_��A�	*O

prediction_loss=
�>

reg_loss��w<


total_loss��>


accuracy_1�z?~�fE]       a[��	��t*_��A�	*O

prediction_loss��?

reg_loss��w<


total_lossy?


accuracy_1���>6M�]       a[��	��t*_��A�	*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?IW�]       a[��	��t*_��A�	*O

prediction_loss=
�>

reg_loss��w<


total_loss��>


accuracy_1�z?po{]       a[��	�u*_��A�	*O

prediction_loss=
�>

reg_loss��w<


total_loss��>


accuracy_1�z?����]       a[��	�"u*_��A�	*O

prediction_loss)\?

reg_loss�w<


total_loss�;?


accuracy_1�G�>zVa]       a[��	<u*_��A�	*O

prediction_loss���>

reg_losst�w<


total_lossc��>


accuracy_1�?B��]       a[��	@Qu*_��A�	*O

prediction_loss   ?

reg_lossi�w<


total_lossj�?


accuracy_1   ?F��]       a[��	�gu*_��A�	*O

prediction_lossq=
?

reg_loss]�w<


total_loss�?


accuracy_1��>��]       a[��	�xu*_��A�	*O

prediction_lossR�?

reg_lossU�w<


total_loss��"?


accuracy_1\��>V�b]       a[��	�u*_��A�	*O

prediction_lossq=
?

reg_lossJ�w<


total_loss�?


accuracy_1��>�O^v]       a[��	��u*_��A�	*O

prediction_loss��?

reg_loss?�w<


total_lossy?


accuracy_1���>�پ]       a[��	�u*_��A�	*O

prediction_lossq=
?

reg_loss3�w<


total_loss�?


accuracy_1��>}���]       a[��	��u*_��A�	*O

prediction_loss{�>

reg_loss+�w<


total_lossLӵ>


accuracy_1��(?�C��]       a[��	��u*_��A�	*O

prediction_loss=
�>

reg_loss �w<


total_loss��>


accuracy_1�z?�3J[]       a[��	Gv*_��A�	*O

prediction_loss   ?

reg_loss�w<


total_lossh�?


accuracy_1   ?A�\`]       a[��	�v*_��A�	*O

prediction_loss��>

reg_loss�w<


total_loss�C�>


accuracy_1q=
?4F��]       a[��	L3v*_��A�	*O

prediction_loss�?

reg_loss�w<


total_loss �?


accuracy_1���>�X�Q]       a[��	�_v*_��A�	*O

prediction_loss��?

reg_loss��w<


total_lossy?


accuracy_1���>H1)]       a[��	M�v*_��A�	*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>����]       a[��	U�v*_��A�	*O

prediction_loss�z?

reg_loss��w<


total_lossIZ?


accuracy_1=
�>�!!�]       a[��	��v*_��A�	*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>o_]       a[��	�w*_��A�	*O

prediction_loss�z?

reg_loss��w<


total_lossHZ?


accuracy_1=
�>i�φ]       a[��	0.w*_��A�	*O

prediction_loss=
�>

reg_loss��w<


total_loss��>


accuracy_1�z?+�GB]       a[��	#Iw*_��A�	*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>���e]       a[��	Aaw*_��A�	*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?!��A]       a[��	�uw*_��A�	*O

prediction_loss���>

reg_loss��w<


total_loss\��>


accuracy_1�?fVw]       a[��	��w*_��A�	*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>��!]       a[��	v�w*_��A�	*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>F�3�]       a[��	��w*_��A�	*O

prediction_loss�z?

reg_loss��w<


total_lossGZ?


accuracy_1=
�>^�]       a[��	��w*_��A�	*O

prediction_loss���>

reg_loss~�w<


total_loss[��>


accuracy_1�?��q�]       a[��	�w*_��A�	*O

prediction_loss��?

reg_losst�w<


total_loss y?


accuracy_1���>M�$]       a[��	p
x*_��A�	*O

prediction_loss�?

reg_lossj�w<


total_loss�?


accuracy_1���>ҋ��]       a[��	�!x*_��A�	*O

prediction_loss��>

reg_loss_�w<


total_loss�C�>


accuracy_1q=
?�$J\]       a[��	�8x*_��A�	*O

prediction_lossq=
?

reg_lossU�w<


total_loss�?


accuracy_1��>ɝ�<]       a[��	�Tx*_��A�	*O

prediction_loss�?

reg_lossJ�w<


total_loss�?


accuracy_1���>T7��]       a[��	px*_��A�	*O

prediction_loss\��>

reg_loss?�w<


total_loss&N�>


accuracy_1R�?���]       a[��	��x*_��A�	*O

prediction_loss��>

reg_loss6�w<


total_loss�C�>


accuracy_1q=
?F�ϯ]       a[��	��x*_��A�	*O

prediction_loss�z?

reg_loss,�w<


total_lossFZ?


accuracy_1=
�>R��]       a[��	ڭx*_��A�	*O

prediction_loss��>

reg_loss!�w<


total_loss�C�>


accuracy_1q=
?�g��]       a[��	��x*_��A�	*O

prediction_loss=
�>

reg_loss�w<


total_loss��>


accuracy_1�z?* ]       a[��	��x*_��A�	*O

prediction_loss   ?

reg_loss�w<


total_lossd�?


accuracy_1   ?���L]       a[��	��x*_��A�	*O

prediction_loss�?

reg_loss�w<


total_loss�?


accuracy_1���>��G5]       a[��	�y*_��A�	*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>N���]       a[��	7y*_��A�	*O

prediction_loss�G�>

reg_loss��w<


total_lossv�>


accuracy_1)\?_�8]       a[��	�1y*_��A�	*O

prediction_loss   ?

reg_loss��w<


total_lossd�?


accuracy_1   ?��d@]       a[��	�Ny*_��A�	*O

prediction_lossR�?

reg_loss��w<


total_loss��"?


accuracy_1\��>͏$�]       a[��	fy*_��A�	*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?_�O�]       a[��	Byy*_��A�	*O

prediction_loss\��>

reg_loss��w<


total_loss"N�>


accuracy_1R�?/�~�]       a[��	��y*_��A�	*O

prediction_loss�G�>

reg_loss��w<


total_losst�>


accuracy_1)\?n�fJ]       a[��	]�y*_��A�	*O

prediction_loss
�#?

reg_loss��w<


total_lossm�'?


accuracy_1�Q�>�^�]       a[��	f�y*_��A�	*O

prediction_loss�z?

reg_loss��w<


total_lossDZ?


accuracy_1=
�>�eW]       a[��	|�y*_��A�	*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?g;c>]       a[��	N�y*_��A�	*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>� �]       a[��	P�y*_��A�	*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?�1"�]       a[��	z*_��A�	*O

prediction_loss�Q�>

reg_loss~�w<


total_loss��>


accuracy_1
�#?���]       a[��	0z*_��A�	*O

prediction_loss��>

reg_lossu�w<


total_loss�C�>


accuracy_1q=
?}�]       a[��	^Hz*_��A�	*O

prediction_loss��>

reg_lossj�w<


total_loss�C�>


accuracy_1q=
?y���]       a[��	�_z*_��A�	*O

prediction_loss
�#?

reg_loss_�w<


total_lossk�'?


accuracy_1�Q�>��},]       a[��	�rz*_��A�	*O

prediction_loss=
�>

reg_lossR�w<


total_loss ��>


accuracy_1�z?Y���]       a[��	��z*_��A�	*O

prediction_loss���>

reg_lossI�w<


total_loss���>


accuracy_1��?[�Z�]       a[��	I�z*_��A�	*O

prediction_loss)\?

reg_loss@�w<


total_loss�;?


accuracy_1�G�>�O=�]       a[��	[�z*_��A�	*O

prediction_loss�G�>

reg_loss5�w<


total_lossp�>


accuracy_1)\?%`�]       a[��	��z*_��A�	*O

prediction_loss��>

reg_loss+�w<


total_loss�C�>


accuracy_1q=
?����]       a[��	��z*_��A�	*O

prediction_loss��>

reg_loss �w<


total_loss�C�>


accuracy_1q=
?3k
]       a[��	��z*_��A�	*O

prediction_loss�G�>

reg_loss�w<


total_losso�>


accuracy_1)\?.&�]       a[��	I{*_��A�	*O

prediction_loss���>

reg_loss�w<


total_lossO��>


accuracy_1�?�Z�]       a[��	 ${*_��A�	*O

prediction_loss���>

reg_loss�w<


total_lossO��>


accuracy_1�?-~��]       a[��	�<{*_��A�	*O

prediction_loss\��>

reg_loss��w<


total_lossN�>


accuracy_1R�?0aI�]       a[��	3P{*_��A�	*O

prediction_lossR�?

reg_loss��w<


total_loss��"?


accuracy_1\��>����]       a[��	�f{*_��A�	*O

prediction_loss   ?

reg_loss��w<


total_loss`�?


accuracy_1   ?�|�]       a[��	�{*_��A�	*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>��K]       a[��	�{*_��A�	*O

prediction_loss�G�>

reg_loss��w<


total_lossl�>


accuracy_1)\?|j��]       a[��	.�{*_��A�	*O

prediction_loss���>

reg_loss��w<


total_lossM��>


accuracy_1�?�k=�]       a[��	��{*_��A�
*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?�T"?]       a[��	r|*_��A�
*O

prediction_loss��?

reg_loss��w<


total_loss�x?


accuracy_1���>���]       a[��	�:|*_��A�
*O

prediction_loss��?

reg_loss��w<


total_loss�x?


accuracy_1���>���]       a[��	R|*_��A�
*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>�|H]       a[��	Di|*_��A�
*O

prediction_loss   ?

reg_loss��w<


total_loss^�?


accuracy_1   ?
�N�]       a[��	˅|*_��A�
*O

prediction_loss   ?

reg_loss��w<


total_loss^�?


accuracy_1   ?�-�i]       a[��	��|*_��A�
*O

prediction_loss���>

reg_lossy�w<


total_lossK��>


accuracy_1�?r{�g]       a[��	��|*_��A�
*O

prediction_loss   ?

reg_lossq�w<


total_loss^�?


accuracy_1   ?����]       a[��	��|*_��A�
*O

prediction_loss�?

reg_lossf�w<


total_loss�?


accuracy_1���>�,f]       a[��	��|*_��A�
*O

prediction_loss=
�>

reg_loss]�w<


total_loss���>


accuracy_1�z?k�]       a[��	Z}*_��A�
*O

prediction_loss   ?

reg_lossQ�w<


total_loss]�?


accuracy_1   ?Ǝ'�]       a[��	9(}*_��A�
*O

prediction_loss�G�>

reg_lossG�w<


total_lossh�>


accuracy_1)\?s��]       a[��	�@}*_��A�
*O

prediction_loss�G�>

reg_loss;�w<


total_lossh�>


accuracy_1)\?��e]       a[��	�T}*_��A�
*O

prediction_loss
�#?

reg_loss2�w<


total_lossg�'?


accuracy_1�Q�>Շ��]       a[��	�j}*_��A�
*O

prediction_lossq=
?

reg_loss(�w<


total_loss�?


accuracy_1��>�N]       a[��	܂}*_��A�
*O

prediction_lossR�?

reg_loss�w<


total_loss��"?


accuracy_1\��>�@P]       a[��	��}*_��A�
*O

prediction_loss   ?

reg_loss�w<


total_loss\�?


accuracy_1   ?Y+`]       a[��	>�}*_��A�
*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>[�� ]       a[��	S�}*_��A�
*O

prediction_loss�G�>

reg_loss��w<


total_lossf�>


accuracy_1)\?��mg]       a[��	@�}*_��A�
*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>�[�]       a[��	�~*_��A�
*O

prediction_loss���>

reg_loss��w<


total_lossF��>


accuracy_1�?�w,]       a[��	Ja~*_��A�
*O

prediction_loss
�#?

reg_loss��w<


total_losse�'?


accuracy_1�Q�>`��]       a[��	_z~*_��A�
*O

prediction_loss���>

reg_loss��w<


total_lossE��>


accuracy_1�?wqx�]       a[��	d�~*_��A�
*O

prediction_loss\��>

reg_loss��w<


total_lossN�>


accuracy_1R�?�%�]       a[��	��~*_��A�
*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�<�]       a[��	-�~*_��A�
*O

prediction_loss�z?

reg_loss��w<


total_loss<Z?


accuracy_1=
�>���]       a[��	O�~*_��A�
*O

prediction_loss   ?

reg_loss��w<


total_loss[�?


accuracy_1   ?�L��]       a[��	K *_��A�
*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>K���]       a[��	]*_��A�
*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?����]       a[��	�.*_��A�
*O

prediction_loss   ?

reg_loss�w<


total_lossZ�?


accuracy_1   ?QF2[]       a[��	D*_��A�
*O

prediction_loss   ?

reg_losss�w<


total_lossZ�?


accuracy_1   ?�eՄ]       a[��	�\*_��A�
*O

prediction_loss=
�>

reg_lossj�w<


total_loss���>


accuracy_1�z?����]       a[��	�q*_��A�
*O

prediction_lossR�?

reg_loss_�w<


total_loss��"?


accuracy_1\��>su �]       a[��	Q�*_��A�
*O

prediction_loss\��>

reg_lossV�w<


total_lossN�>


accuracy_1R�?���&]       a[��	��*_��A�
*O

prediction_loss�?

reg_lossM�w<


total_loss�?


accuracy_1���>�p�9]       a[��	�*_��A�
*O

prediction_loss��?

reg_lossC�w<


total_loss�x?


accuracy_1���>�Y�]       a[��	|�*_��A�
*O

prediction_loss)\?

reg_loss9�w<


total_loss�;?


accuracy_1�G�>���]       a[��	��*_��A�
*O

prediction_lossq=
?

reg_loss/�w<


total_loss�?


accuracy_1��>	�%�]       a[��	p�*_��A�
*O

prediction_loss��?

reg_loss&�w<


total_loss�x?


accuracy_1���>��]       a[��	�!�*_��A�
*O

prediction_loss���>

reg_loss�w<


total_loss~��>


accuracy_1��?�6�E]       a[��	�8�*_��A�
*O

prediction_loss���>

reg_loss�w<


total_loss~��>


accuracy_1��?���]       a[��	�[�*_��A�
*O

prediction_loss�G�>

reg_loss�w<


total_loss^�>


accuracy_1)\?� ��]       a[��	��*_��A�
*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>~gd>]       a[��	ͫ�*_��A�
*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>�.�]       a[��	�ʀ*_��A�
*O

prediction_loss�G�>

reg_loss��w<


total_loss]�>


accuracy_1)\??o��]       a[��		�*_��A�
*O

prediction_loss)\?

reg_loss��w<


total_loss�;?


accuracy_1�G�>e��]       a[��	�&�*_��A�
*O

prediction_loss�G�>

reg_loss��w<


total_loss]�>


accuracy_1)\?��44]       a[��	�H�*_��A�
*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?0.bN]       a[��	ke�*_��A�
*O

prediction_loss���>

reg_loss��w<


total_loss=��>


accuracy_1�?�O�]       a[��	t~�*_��A�
*O

prediction_loss�G�>

reg_loss��w<


total_loss\�>


accuracy_1)\?�{�]       a[��	���*_��A�
*O

prediction_loss\��>

reg_loss��w<


total_loss	N�>


accuracy_1R�?�|��]       a[��	з�*_��A�
*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>h�	�]       a[��	=Ձ*_��A�
*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?-/��]       a[��	��*_��A�
*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>Љ�]       a[��	��*_��A�
*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>��8]       a[��	:!�*_��A�
*O

prediction_loss   ?

reg_lossx�w<


total_lossV�?


accuracy_1   ?��R�]       a[��	�9�*_��A�
*O

prediction_loss{�>

reg_lossl�w<


total_loss&ӵ>


accuracy_1��(?u��^]       a[��	�Q�*_��A�
*O

prediction_loss��?

reg_lossd�w<


total_loss�x?


accuracy_1���>����]       a[��	{j�*_��A�
*O

prediction_lossq=
?

reg_lossY�w<


total_loss�?


accuracy_1��>��h�]       a[��	���*_��A�
*O

prediction_loss��>

reg_lossP�w<


total_loss�C�>


accuracy_1q=
?FW��]       a[��	˜�*_��A�
*O

prediction_loss   ?

reg_lossF�w<


total_lossU�?


accuracy_1   ?N���]       a[��	���*_��A�
*O

prediction_loss
�#?

reg_loss;�w<


total_loss_�'?


accuracy_1�Q�>�l��]       a[��	1͂*_��A�
*O

prediction_loss{�>

reg_loss/�w<


total_loss$ӵ>


accuracy_1��(?��3�]       a[��	;�*_��A�
*O

prediction_loss)\?

reg_loss%�w<


total_loss~;?


accuracy_1�G�>���]       a[��	��*_��A�
*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>���]       a[��	�*_��A�
*O

prediction_loss�?

reg_loss�w<


total_loss�?


accuracy_1���>m+�]       a[��	80�*_��A�
*O

prediction_loss�?

reg_loss�w<


total_loss�?


accuracy_1���>���+]       a[��	<J�*_��A�
*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>pW��]       a[��	�a�*_��A�
*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?e�]       a[��	N��*_��A�
*O

prediction_loss
�#?

reg_loss��w<


total_loss^�'?


accuracy_1�Q�>�P�]       a[��	E؃*_��A�
*O

prediction_loss   ?

reg_loss��w<


total_lossT�?


accuracy_1   ?:�߽]       a[��	H��*_��A�
*O

prediction_loss   ?

reg_loss��w<


total_lossS�?


accuracy_1   ?�F�3]       a[��	F�*_��A�
*O

prediction_loss)\?

reg_loss��w<


total_loss|;?


accuracy_1�G�>?#]       a[��	�w�*_��A�
*O

prediction_loss���>

reg_loss��w<


total_loss5��>


accuracy_1�?�E%]       a[��	p��*_��A�
*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>
�uf]       a[��	�*_��A�
*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>���_]       a[��	�ބ*_��A�
*O

prediction_loss{.?

reg_loss��w<


total_loss��1?


accuracy_1
ף>a2�]       a[��	�	�*_��A�
*O

prediction_loss)\?

reg_loss��w<


total_loss{;?


accuracy_1�G�>D�w�]       a[��	'O�*_��A�
*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�ȗ�]       a[��	t{�*_��A�
*O

prediction_loss���>

reg_loss��w<


total_loss3��>


accuracy_1�?
��]       a[��	���*_��A�
*O

prediction_loss)\?

reg_lossu�w<


total_loss{;?


accuracy_1�G�> �u ]       a[��	�̅*_��A�
*O

prediction_loss\��>

reg_lossk�w<


total_loss�M�>


accuracy_1R�?1UE]]       a[��	��*_��A�
*O

prediction_loss�?

reg_loss`�w<


total_loss
�?


accuracy_1���>ї��]       a[��	j/�*_��A�
*O

prediction_loss�Q�>

reg_lossY�w<


total_loss��>


accuracy_1
�#?9U�]       a[��	�T�*_��A�
*O

prediction_loss   ?

reg_lossN�w<


total_lossQ�?


accuracy_1   ?���e]       a[��	���*_��A�
*O

prediction_loss���>

reg_lossD�w<


total_loss1��>


accuracy_1�?}ȁ]       a[��	{��*_��A�
*O

prediction_loss   ?

reg_loss9�w<


total_lossQ�?


accuracy_1   ?v��]       a[��	*ʆ*_��A�
*O

prediction_lossq=
?

reg_loss0�w<


total_loss�?


accuracy_1��>Z{�+]       a[��	�	�*_��A�
*O

prediction_lossq=
?

reg_loss$�w<


total_loss�?


accuracy_1��>+��9]       a[��	�,�*_��A�
*O

prediction_loss   ?

reg_loss�w<


total_lossP�?


accuracy_1   ?L��]       a[��	�e�*_��A�
*O

prediction_loss�?

reg_loss�w<


total_loss�?


accuracy_1���>��]       a[��	���*_��A�
*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>a��b]       a[��	6��*_��A�
*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?v@$�]       a[��	�̇*_��A�
*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?QW�]       a[��	B�*_��A�
*O

prediction_loss���>

reg_loss��w<


total_loss.��>


accuracy_1�?N��]       a[��	G!�*_��A�
*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?C��]       a[��	�Y�*_��A�
*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>F�k�]       a[��	��*_��A�
*O

prediction_loss   ?

reg_loss��w<


total_lossO�?


accuracy_1   ?>b�]       a[��	��*_��A�
*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?
�*�]       a[��	�ň*_��A�
*O

prediction_loss�G�>

reg_loss��w<


total_lossK�>


accuracy_1)\?�<�]       a[��	��*_��A�
*O

prediction_loss�G�>

reg_loss��w<


total_lossK�>


accuracy_1)\?�\�]       a[��	^�*_��A�
*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?p�A]       a[��	U0�*_��A�
*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>K�{]       a[��	D��*_��A�
*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�VW]       a[��	}��*_��A�
*O

prediction_loss�z?

reg_lossz�w<


total_loss/Z?


accuracy_1=
�>C�@]       a[��	RӉ*_��A�
*O

prediction_loss���>

reg_lossr�w<


total_lossi��>


accuracy_1��?G6Yg]       a[��	
��*_��A�
*O

prediction_loss���>

reg_lossg�w<


total_loss*��>


accuracy_1�?�9N�]       a[��	��*_��A�
*O

prediction_loss=
�>

reg_loss\�w<


total_loss���>


accuracy_1�z?GH�]       a[��	p[�*_��A�
*O

prediction_loss�?

reg_lossR�w<


total_loss�?


accuracy_1���>ʦ�]       a[��	T��*_��A�
*O

prediction_loss{�>

reg_lossI�w<


total_lossӵ>


accuracy_1��(?���]       a[��	ٴ�*_��A�
*O

prediction_loss�z?

reg_loss=�w<


total_loss.Z?


accuracy_1=
�>�>��]       a[��	Y��*_��A�
*O

prediction_loss�?

reg_loss3�w<


total_loss�?


accuracy_1���>"���]       a[��	�*_��A�
*O

prediction_loss�?

reg_loss*�w<


total_loss�?


accuracy_1���>|O�]       a[��	f2�*_��A�
*O

prediction_loss��?

reg_loss�w<


total_loss�x?


accuracy_1���>͏�5]       a[��	Vb�*_��A�
*O

prediction_loss�?

reg_loss�w<


total_loss�?


accuracy_1���>���1]       a[��	��*_��A�
*O

prediction_loss   ?

reg_loss	�w<


total_lossL�?


accuracy_1   ?E�1]       a[��	��*_��A�
*O

prediction_lossR�?

reg_loss �w<


total_loss��"?


accuracy_1\��>�d]       a[��	��*_��A�
*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?X�j�]       a[��	L5�*_��A�
*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?���]       a[��	�q�*_��A�
*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>G��]       a[��	I��*_��A�
*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>F���]       a[��	g~�*_��A�
*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>��]       a[��	���*_��A�
*O

prediction_loss���>

reg_loss��w<


total_lossc��>


accuracy_1��?JT$#]       a[��	5ҍ*_��A�
*O

prediction_loss\��>

reg_loss��w<


total_loss�M�>


accuracy_1R�?�|��]       a[��	���*_��A�
*O

prediction_loss�G�>

reg_loss��w<


total_lossC�>


accuracy_1)\?��]       a[��	��*_��A�
*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�:��]       a[��	�8�*_��A�
*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>'�Gi]       a[��	�^�*_��A�
*O

prediction_loss)\?

reg_loss��w<


total_losss;?


accuracy_1�G�>#U�O]       a[��	��*_��A�*O

prediction_loss��?

reg_lossz�w<


total_loss�x?


accuracy_1���>9+6]       a[��	�Î*_��A�*O

prediction_loss��>

reg_lossq�w<


total_loss�C�>


accuracy_1q=
?%�_]       a[��	��*_��A�*O

prediction_loss\��>

reg_lossf�w<


total_loss�M�>


accuracy_1R�?2��]       a[��	z4�*_��A�*O

prediction_loss���>

reg_loss]�w<


total_loss`��>


accuracy_1��?Q5A�]       a[��	��*_��A�*O

prediction_loss)\?

reg_lossP�w<


total_lossr;?


accuracy_1�G�>
�p]       a[��	qǏ*_��A�*O

prediction_loss�?

reg_lossF�w<


total_loss�?


accuracy_1���>�>h]       a[��	/��*_��A�*O

prediction_loss\��>

reg_loss<�w<


total_loss�M�>


accuracy_1R�?�7]       a[��	U�*_��A�*O

prediction_loss�z?

reg_loss2�w<


total_loss*Z?


accuracy_1=
�>�i�]       a[��	���*_��A�*O

prediction_loss��>

reg_loss(�w<


total_loss�C�>


accuracy_1q=
?2�`�]       a[��	�ɐ*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss �?


accuracy_1���>H��0]       a[��	P��*_��A�*O

prediction_loss�Q�>

reg_loss�w<


total_loss}�>


accuracy_1
�#?�iw]       a[��	B%�*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss>�>


accuracy_1)\?��]       a[��	NC�*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>fb߰]       a[��	^�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss �?


accuracy_1���>M]}�]       a[��	���*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossq;?


accuracy_1�G�>��E]       a[��	�ő*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?JϿ]       a[��	}�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss\��>


accuracy_1��?�c�]       a[��	[�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss<�>


accuracy_1)\?T��]       a[��	r3�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>�0�]       a[��	�\�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?�Ř�]       a[��	~��*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?��0]       a[��	C��*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss;�>


accuracy_1)\?�;]       a[��	�ǒ*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?��']       a[��	ޒ*_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�M�>


accuracy_1R�?�~�K]       a[��	J	�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?��]       a[��	�+�*_��A�*O

prediction_lossq=
?

reg_lossw�w<


total_loss�?


accuracy_1��>̛]       a[��	ZE�*_��A�*O

prediction_loss=
�>

reg_losso�w<


total_loss���>


accuracy_1�z?c2��]       a[��	�^�*_��A�*O

prediction_loss���>

reg_lossb�w<


total_loss��>


accuracy_1�?]�|]       a[��	�v�*_��A�*O

prediction_loss   ?

reg_lossY�w<


total_lossE�?


accuracy_1   ?�WU�]       a[��	6��*_��A�*O

prediction_lossq=
?

reg_lossN�w<


total_loss�?


accuracy_1��>]�Q�]       a[��	-��*_��A�*O

prediction_lossR�?

reg_lossD�w<


total_loss��"?


accuracy_1\��>AxP�]       a[��	>˓*_��A�*O

prediction_loss���>

reg_loss;�w<


total_loss��>


accuracy_1�?Tʝ*]       a[��	��*_��A�*O

prediction_loss�z?

reg_loss0�w<


total_loss&Z?


accuracy_1=
�>��]       a[��	���*_��A�*O

prediction_loss�z?

reg_loss&�w<


total_loss&Z?


accuracy_1=
�>�_K]       a[��	��*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>>8�]       a[��	1�*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>��m]       a[��	�I�*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?�-!v]       a[��	R`�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossm;?


accuracy_1�G�>�o,]       a[��	�w�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossD�?


accuracy_1   ?�Hd,]       a[��	���*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?�P]       a[��	Ƨ�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?G��]       a[��	���*_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossC�?


accuracy_1   ?8��]       a[��	Qݔ*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>]���]       a[��	I��*_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossC�?


accuracy_1   ?e/�]       a[��	f�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>R�T]       a[��	�1�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>g�]       a[��	�J�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�V�]       a[��	Rb�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossB�?


accuracy_1   ?,1]       a[��	~�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?��P�]       a[��	̙�*_��A�*O

prediction_loss)\?

reg_loss~�w<


total_lossk;?


accuracy_1�G�>�o��]       a[��	y��*_��A�*O

prediction_loss   ?

reg_lossu�w<


total_lossB�?


accuracy_1   ?�0Ce]       a[��	ɕ*_��A�*O

prediction_loss=
�>

reg_lossk�w<


total_loss���>


accuracy_1�z?�/�]       a[��	�ߕ*_��A�*O

prediction_loss���>

reg_lossa�w<


total_lossP��>


accuracy_1��?�u4v]       a[��	���*_��A�*O

prediction_loss���>

reg_lossX�w<


total_lossP��>


accuracy_1��?L!��]       a[��	j�*_��A�*O

prediction_loss�?

reg_lossN�w<


total_loss��?


accuracy_1���>ݱN]       a[��	0�*_��A�*O

prediction_loss��>

reg_lossB�w<


total_loss�C�>


accuracy_1q=
?���	]       a[��	�G�*_��A�*O

prediction_loss   ?

reg_loss8�w<


total_lossA�?


accuracy_1   ?�p�N]       a[��	�`�*_��A�*O

prediction_loss���>

reg_loss/�w<


total_loss��>


accuracy_1�?e��}]       a[��	�x�*_��A�*O

prediction_loss=
�>

reg_loss$�w<


total_loss���>


accuracy_1�z?|�*x]       a[��	�*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss@�?


accuracy_1   ?<C�]       a[��	��*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss.�>


accuracy_1)\?��[]       a[��	$��*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss@�?


accuracy_1   ?:�Nq]       a[��	�Ж*_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss��"?


accuracy_1\��>�E2J]       a[��	h�*_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�M�>


accuracy_1R�?��]       a[��	��*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?��b6]       a[��	��*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss Z?


accuracy_1=
�>���]       a[��	�4�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss Z?


accuracy_1=
�>J��]       a[��	�K�*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?� ]       a[��	�d�*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?Y �r]       a[��	 {�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss,�>


accuracy_1)\?<��]       a[��	>��*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�Dc�]       a[��	Ҭ�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�R�7]       a[��	.ȗ*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossg;?


accuracy_1�G�>:��]       a[��	Yߗ*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss*�>


accuracy_1)\?k��$]       a[��	���*_��A�*O

prediction_loss)\?

reg_loss|�w<


total_lossg;?


accuracy_1�G�>�]       a[��	��*_��A�*O

prediction_lossq=
?

reg_lossr�w<


total_loss�?


accuracy_1��>d�Y]       a[��	�%�*_��A�*O

prediction_lossq=
?

reg_lossh�w<


total_loss�?


accuracy_1��>73�]       a[��	S@�*_��A�*O

prediction_loss��>

reg_loss`�w<


total_loss�C�>


accuracy_1q=
?���u]       a[��	yW�*_��A�*O

prediction_loss��>

reg_lossV�w<


total_loss�C�>


accuracy_1q=
?rn(�]       a[��	Pn�*_��A�*O

prediction_lossq=
?

reg_lossJ�w<


total_loss�?


accuracy_1��>�P�j]       a[��	��*_��A�*O

prediction_loss   ?

reg_loss@�w<


total_loss=�?


accuracy_1   ?�YV<]       a[��	ᛘ*_��A�*O

prediction_loss)\?

reg_loss7�w<


total_lossf;?


accuracy_1�G�>���z]       a[��	k��*_��A�*O

prediction_loss��>

reg_loss-�w<


total_loss�C�>


accuracy_1q=
?*؀l]       a[��	�ј*_��A�*O

prediction_loss)\?

reg_loss!�w<


total_lossf;?


accuracy_1�G�>�%�]       a[��	d�*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?.���]       a[��	7��*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss&�>


accuracy_1)\?ȁ7�]       a[��	�*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�C�>


accuracy_1q=
?��?�]       a[��	I.�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss<�?


accuracy_1   ?$]�A]       a[��	ZJ�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?�l!!]       a[��	fk�*_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss��"?


accuracy_1\��>�Uw]       a[��	w��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>܂']       a[��	���*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�x?


accuracy_1���>���]       a[��	��*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?�p~�]       a[��	���*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossd;?


accuracy_1�G�>���3]       a[��	��*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss;�?


accuracy_1   ?R�]       a[��	�.�*_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss��"?


accuracy_1\��>G��`]       a[��	�P�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?]��]       a[��	�p�*_��A�*O

prediction_loss��(?

reg_loss��w<


total_loss��,?


accuracy_1{�>�2�[]       a[��	
��*_��A�*O

prediction_loss   ?

reg_lossy�w<


total_loss:�?


accuracy_1   ?J��J]       a[��	���*_��A�*O

prediction_loss)\?

reg_lossq�w<


total_lossc;?


accuracy_1�G�>���F]       a[��	9֚*_��A�*O

prediction_loss\��>

reg_lossf�w<


total_loss�M�>


accuracy_1R�?�B��]       a[��	 �*_��A�*O

prediction_loss���>

reg_loss[�w<


total_loss��>


accuracy_1�?V���]       a[��	��*_��A�*O

prediction_loss)\?

reg_lossP�w<


total_lossb;?


accuracy_1�G�>�.��]       a[��	�*_��A�*O

prediction_loss)\?

reg_lossF�w<


total_lossb;?


accuracy_1�G�>��&\]       a[��	5�*_��A�*O

prediction_loss��>

reg_loss;�w<


total_loss�C�>


accuracy_1q=
?ء��]       a[��	O�*_��A�*O

prediction_loss���>

reg_loss2�w<


total_loss��>


accuracy_1�?���]       a[��	i�*_��A�*O

prediction_loss
�#?

reg_loss'�w<


total_lossC�'?


accuracy_1�Q�>ɺ�]       a[��	o��*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss8�?


accuracy_1   ?j�`]       a[��	;��*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>��m]       a[��	�Л*_��A�*O

prediction_loss
�#?

reg_loss�w<


total_lossB�'?


accuracy_1�Q�>��Y]       a[��	|�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss8�?


accuracy_1   ?FHj]       a[��	F%�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossa;?


accuracy_1�G�>
4��]       a[��	�J�*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�ü]       a[��	/n�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>O�^]       a[��	㌜*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>gU�]       a[��	��*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�]]       a[��	~�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>IJ]       a[��	�4�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?ߪ��]       a[��	�K�*_��A�*O

prediction_loss{�>

reg_loss��w<


total_loss�ҵ>


accuracy_1��(?�=]       a[��	8h�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?_��]       a[��	䁝*_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossZ?


accuracy_1=
�>�sV]       a[��	�*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�x?


accuracy_1���>�{��]       a[��	$��*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss6�?


accuracy_1   ?�R��]       a[��	�Н*_��A�*O

prediction_loss)\?

reg_lossx�w<


total_loss_;?


accuracy_1�G�>���]       a[��	y�*_��A�*O

prediction_loss���>

reg_lossl�w<


total_loss���>


accuracy_1�?%,"�]       a[��	2�*_��A�*O

prediction_loss���>

reg_lossb�w<


total_loss���>


accuracy_1�?�u*�]       a[��	��*_��A�*O

prediction_loss���>

reg_lossZ�w<


total_loss���>


accuracy_1�?�p��]       a[��	+2�*_��A�*O

prediction_loss���>

reg_lossN�w<


total_loss���>


accuracy_1�?�9z�]       a[��	|H�*_��A�*O

prediction_loss�G�>

reg_lossC�w<


total_loss�>


accuracy_1)\?P��K]       a[��	c�*_��A�*O

prediction_loss   ?

reg_loss9�w<


total_loss5�?


accuracy_1   ?�E��]       a[��	�y�*_��A�*O

prediction_loss   ?

reg_loss0�w<


total_loss5�?


accuracy_1   ?�$�]       a[��	���*_��A�*O

prediction_loss�?

reg_loss%�w<


total_loss��?


accuracy_1���>|c��]       a[��	��*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�C�>


accuracy_1q=
?�|%�]       a[��	���*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>���]       a[��	�֞*_��A�*O

prediction_loss
�#?

reg_loss�w<


total_loss>�'?


accuracy_1�Q�>�6k]       a[��	���*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�C�>


accuracy_1q=
?<X�]       a[��	��*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?4}�]       a[��	-!�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss�>


accuracy_1)\?X�]       a[��	]7�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss\;?


accuracy_1�G�>���]       a[��	�K�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>��]       a[��	Eb�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>L�/�]       a[��	pz�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?�g	
]       a[��	���*_��A�*O

prediction_loss{.?

reg_loss��w<


total_loss��1?


accuracy_1
ף>7��C]       a[��	ʤ�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss�>


accuracy_1)\?��k]       a[��	#��*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�B�]       a[��	Ο*_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossZ?


accuracy_1=
�>�>ʣ]       a[��	��*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss2�?


accuracy_1   ?�}�A]       a[��	��*_��A�*O

prediction_loss333?

reg_loss~�w<


total_losse7?


accuracy_1���>���E]       a[��	�$�*_��A�*O

prediction_loss�G�>

reg_lossu�w<


total_loss�>


accuracy_1)\?�~�X]       a[��	P:�*_��A�*O

prediction_loss��?

reg_lossj�w<


total_loss�x?


accuracy_1���>ͬ�]       a[��	*R�*_��A�*O

prediction_loss��>

reg_lossb�w<


total_loss�C�>


accuracy_1q=
?D��]       a[��	Kv�*_��A�*O

prediction_loss�Q�>

reg_lossV�w<


total_lossO�>


accuracy_1
�#?��,]       a[��	���*_��A�*O

prediction_loss��>

reg_lossN�w<


total_loss�C�>


accuracy_1q=
?6���]       a[��	���*_��A�*O

prediction_loss=
�>

reg_lossC�w<


total_loss���>


accuracy_1�z?�l�]       a[��	��*_��A�*O

prediction_loss=
�>

reg_loss9�w<


total_loss���>


accuracy_1�z?����]       a[��	�A�*_��A�*O

prediction_loss�?

reg_loss.�w<


total_loss��?


accuracy_1���>*5r�]       a[��	�^�*_��A�*O

prediction_lossq=
?

reg_loss$�w<


total_loss�?


accuracy_1��>��X�]       a[��	���*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1�?��͢]       a[��	�á*_��A�*O

prediction_loss��>

reg_loss�w<


total_lossC�>


accuracy_1q=
?;�&�]       a[��	��*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>��0v]       a[��	��*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>��e�]       a[��	3�*_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�M�>


accuracy_1R�?@��]       a[��	0g�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�Oj?]       a[��	ɒ�*_��A�*O

prediction_loss{�>

reg_loss��w<


total_loss�ҵ>


accuracy_1��(?c���]       a[��	���*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss~C�>


accuracy_1q=
?���K]       a[��	�ޢ*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss/�?


accuracy_1   ?�z��]       a[��	�F�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>��J]       a[��	�o�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?,�3�]       a[��	��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>���]       a[��	'��*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss.�?


accuracy_1   ?�x��]       a[��	^�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>R2�
]       a[��	C8�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss{C�>


accuracy_1q=
?O�)�]       a[��	WZ�*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss.�?


accuracy_1   ?�$�]       a[��	H��*_��A�*O

prediction_loss=
�>

reg_lossw�w<


total_loss���>


accuracy_1�z?�a;]       a[��	ɤ*_��A�*O

prediction_loss��>

reg_lossj�w<


total_losszC�>


accuracy_1q=
?�B�]       a[��	��*_��A�*O

prediction_loss)\?

reg_lossa�w<


total_lossW;?


accuracy_1�G�>?=-r]       a[��	-�*_��A�*O

prediction_loss��>

reg_lossX�w<


total_losszC�>


accuracy_1q=
?��Am]       a[��	�U�*_��A�*O

prediction_loss�G�>

reg_lossM�w<


total_loss�>


accuracy_1)\?6��3]       a[��	Y��*_��A�*O

prediction_loss)\?

reg_lossC�w<


total_lossV;?


accuracy_1�G�>rҜ�]       a[��	l̥*_��A�*O

prediction_loss�z?

reg_loss8�w<


total_lossZ?


accuracy_1=
�>JT�]       a[��	���*_��A�*O

prediction_loss)\?

reg_loss,�w<


total_lossV;?


accuracy_1�G�>�:�h]       a[��	2�*_��A�*O

prediction_loss�z?

reg_loss"�w<


total_lossZ?


accuracy_1=
�>4�]       a[��	�x�*_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?9�L�]       a[��	p��*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>f�Y�]       a[��	���*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>	�h]       a[��	w��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>�G{�]       a[��	�*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�_�d]       a[��	�X�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossU;?


accuracy_1�G�>$���]       a[��	�z�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss+�?


accuracy_1   ?�)P]       a[��	���*_��A�*O

prediction_loss{�>

reg_loss��w<


total_loss�ҵ>


accuracy_1��(?�C�+]       a[��	ʧ*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�Lk�]       a[��	���*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossuC�>


accuracy_1q=
?�٥]       a[��	Z�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossZ?


accuracy_1=
�>�ڤL]       a[��	�&�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�V�]       a[��	K�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossS;?


accuracy_1�G�>B�(�]       a[��	�p�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossS;?


accuracy_1�G�>��ܷ]       a[��	h��*_��A�*O

prediction_loss��?

reg_loss~�w<


total_loss�x?


accuracy_1���>��Z�]       a[��	�թ*_��A�*O

prediction_loss   ?

reg_losst�w<


total_loss*�?


accuracy_1   ?���]       a[��	��*_��A�*O

prediction_loss)\?

reg_lossi�w<


total_lossS;?


accuracy_1�G�>Uh��]       a[��	kH�*_��A�*O

prediction_loss�Q�>

reg_loss`�w<


total_loss?�>


accuracy_1
�#?�F��]       a[��	7l�*_��A�*O

prediction_loss   ?

reg_lossS�w<


total_loss)�?


accuracy_1   ?B�Q�]       a[��	ʍ�*_��A�*O

prediction_loss\��>

reg_lossJ�w<


total_loss�M�>


accuracy_1R�?�"��]       a[��	�*_��A�*O

prediction_lossq=
?

reg_loss@�w<


total_loss�?


accuracy_1��>����]       a[��	['�*_��A�*O

prediction_loss=
�>

reg_loss7�w<


total_loss���>


accuracy_1�z?I`]       a[��	�_�*_��A�*O

prediction_loss���>

reg_loss)�w<


total_loss��>


accuracy_1��?��H�]       a[��	t��*_��A�*O

prediction_loss�?

reg_loss!�w<


total_loss��?


accuracy_1���>��g�]       a[��	�׫*_��A�*O

prediction_loss��>

reg_loss�w<


total_losspC�>


accuracy_1q=
?��w]       a[��	h>�*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?�Z�z]       a[��	���*_��A�*O

prediction_lossR�?

reg_loss�w<


total_lossz�"?


accuracy_1\��>�Ќ�]       a[��	n�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss(�?


accuracy_1   ?N�^]       a[��	l�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>}ƹX]       a[��	A,�*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossހ�>


accuracy_1�?JCgY]       a[��	*X�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss'�?


accuracy_1   ?0� 3]       a[��	(��*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�׀�]       a[��	ۭ*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?��ݚ]       a[��	� �*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss݀�>


accuracy_1�?�G
]       a[��	u!�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1��?���]       a[��	YP�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossP;?


accuracy_1�G�>��JP]       a[��	˜�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>2sב]       a[��	6ʮ*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>��E�]       a[��	��*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss8�>


accuracy_1
�#?P�6]       a[��	�.�*_��A�*O

prediction_lossq=
?

reg_loss{�w<


total_loss�?


accuracy_1��>T\�|]       a[��	�v�*_��A�*O

prediction_loss��?

reg_lossr�w<


total_loss�x?


accuracy_1���>�Dy3]       a[��	囯*_��A�*O

prediction_loss��>

reg_lossf�w<


total_lossjC�>


accuracy_1q=
?�9��]       a[��	�ǯ*_��A�*O

prediction_loss{.?

reg_loss\�w<


total_loss��1?


accuracy_1
ף>��X�]       a[��	��*_��A�*O

prediction_loss�G�>

reg_lossP�w<


total_loss��>


accuracy_1)\?�h�]       a[��	kH�*_��A�*O

prediction_loss   ?

reg_lossG�w<


total_loss%�?


accuracy_1   ?o�+{]       a[��	��*_��A�*O

prediction_lossq=
?

reg_loss<�w<


total_loss�?


accuracy_1��>�k�7]       a[��	a��*_��A�*O

prediction_loss��?

reg_loss2�w<


total_loss�x?


accuracy_1���>�
�]       a[��	�װ*_��A�*O

prediction_loss
�#?

reg_loss(�w<


total_loss/�'?


accuracy_1�Q�>eGz]       a[��	P7�*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss؀�>


accuracy_1�?���]       a[��	)\�*_��A�*O

prediction_loss�z?

reg_loss�w<


total_lossZ?


accuracy_1=
�>���]       a[��	N�*_��A�*O

prediction_loss)\?

reg_loss
�w<


total_lossM;?


accuracy_1�G�>���]       a[��	<��*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss׀�>


accuracy_1�?���]       a[��	׿�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossM;?


accuracy_1�G�>�Ƀ]       a[��	&�*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossfC�>


accuracy_1q=
?hpн]       a[��	�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>����]       a[��	�(�*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossfC�>


accuracy_1q=
?�6��]       a[��	K�*_��A�*O

prediction_loss��>

reg_loss��w<


total_losseC�>


accuracy_1q=
?W�V`]       a[��	Pq�*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossՀ�>


accuracy_1�?s��.]       a[��	!��*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1��?Q��V]       a[��	氲*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?Zr{P]       a[��	�Ӳ*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss#�?


accuracy_1   ?w�ɵ]       a[��	;�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>ȅ��]       a[��	~�*_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�M�>


accuracy_1R�?��]       a[��	�:�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?K�E�]       a[��	\Z�*_��A�*O

prediction_loss��?

reg_lossy�w<


total_loss�x?


accuracy_1���>Sf��]       a[��	��*_��A�*O

prediction_loss)\?

reg_losso�w<


total_lossK;?


accuracy_1�G�>�Z��]       a[��	ک�*_��A�*O

prediction_loss��>

reg_losse�w<


total_lossbC�>


accuracy_1q=
?jw]]       a[��	�˳*_��A�*O

prediction_loss��>

reg_lossZ�w<


total_lossbC�>


accuracy_1q=
?ǖ��]       a[��	��*_��A�*O

prediction_loss�?

reg_lossN�w<


total_loss��?


accuracy_1���>zT��]       a[��	��*_��A�*O

prediction_loss�z?

reg_lossF�w<


total_lossZ?


accuracy_1=
�>QQ�!]       a[��	B�*_��A�*O

prediction_loss���>

reg_loss;�w<


total_lossр�>


accuracy_1�?�ٛX]       a[��	Vd�*_��A�*O

prediction_loss���>

reg_loss0�w<


total_lossЀ�>


accuracy_1�?���]       a[��	{��*_��A�*O

prediction_loss��?

reg_loss(�w<


total_loss�x?


accuracy_1���>�#]       a[��	���*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss �?


accuracy_1   ?|�~]       a[��	K̴*_��A�*O

prediction_loss{.?

reg_loss�w<


total_loss��1?


accuracy_1
ף>�]       a[��	��*_��A�*O

prediction_loss�G�>

reg_loss	�w<


total_loss��>


accuracy_1)\?;]       a[��	��*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossI;?


accuracy_1�G�>�N[]       a[��	�9�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss_C�>


accuracy_1q=
?�)�]       a[��	.W�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss΀�>


accuracy_1�?<.��]       a[��	�z�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss^C�>


accuracy_1q=
?4唰]       a[��	ˢ�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss Z?


accuracy_1=
�>j���]       a[��	{��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>9��Q]       a[��	kص*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss̀�>


accuracy_1�?�;֫]       a[��	���*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossH;?


accuracy_1�G�>��I�]       a[��	��*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>~�-�]       a[��	�B�*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossz��>


accuracy_1�z?oHg
]       a[��	�]�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss̀�>


accuracy_1�?2��]       a[��	lz�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�d�]       a[��	�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>n�4B]       a[��	��*_��A�*O

prediction_loss   ?

reg_lossx�w<


total_loss�?


accuracy_1   ?>��Q]       a[��	�ڶ*_��A�*O

prediction_loss��>

reg_lossl�w<


total_lossZC�>


accuracy_1q=
?�˸�]       a[��	'��*_��A�*O

prediction_lossq=
?

reg_lossc�w<


total_loss�?


accuracy_1��>�KA�]       a[��	�*_��A�*O

prediction_loss�?

reg_lossX�w<


total_loss��?


accuracy_1���>��ly]       a[��	�?�*_��A�*O

prediction_loss)\?

reg_lossO�w<


total_lossF;?


accuracy_1�G�>�q:6]       a[��	>\�*_��A�*O

prediction_loss�?

reg_lossE�w<


total_loss��?


accuracy_1���>K�®]       a[��	.t�*_��A�*O

prediction_loss)\?

reg_loss;�w<


total_lossF;?


accuracy_1�G�>�÷�]       a[��	���*_��A�*O

prediction_loss���>

reg_loss/�w<


total_lossȀ�>


accuracy_1�?S��]       a[��	غ�*_��A�*O

prediction_lossq=
?

reg_loss&�w<


total_loss�?


accuracy_1��>m/��]       a[��	�ܷ*_��A�*O

prediction_loss��>

reg_loss�w<


total_lossXC�>


accuracy_1q=
?��^�]       a[��	C��*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>�~/�]       a[��	�!�*_��A�*O

prediction_loss)\?

reg_loss�w<


total_lossE;?


accuracy_1�G�>E�B]       a[��	YN�*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossWC�>


accuracy_1q=
?I�l�]       a[��	�p�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>
�Ϗ]       a[��	�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?����]       a[��	��*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossD;?


accuracy_1�G�>#4�]       a[��	�ܸ*_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossD;?


accuracy_1�G�>\�Q]       a[��	��*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossUC�>


accuracy_1q=
?�O�D]       a[��	��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>,��]       a[��	_B�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?�Qd�]       a[��	�b�*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossTC�>


accuracy_1q=
?9O@]       a[��	r��*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossĀ�>


accuracy_1�?���&]       a[��	��*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�x?


accuracy_1���>��:]       a[��	_&�*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>=ͽ�]       a[��	>�*_��A�*O

prediction_loss�?

reg_lossw�w<


total_loss��?


accuracy_1���>zA��]       a[��	�[�*_��A�*O

prediction_lossq=
?

reg_lossl�w<


total_loss�?


accuracy_1��>��]]       a[��	k}�*_��A�*O

prediction_loss   ?

reg_lossa�w<


total_loss�?


accuracy_1   ?0�|]       a[��	8��*_��A�*O

prediction_loss��>

reg_lossW�w<


total_lossRC�>


accuracy_1q=
?z�>�]       a[��	�ź*_��A�*O

prediction_loss�z?

reg_lossL�w<


total_loss�Y?


accuracy_1=
�>�ZI�]       a[��	��*_��A�*O

prediction_loss=
�>

reg_lossC�w<


total_losso��>


accuracy_1�z?�v�]       a[��	|�*_��A�*O

prediction_loss���>

reg_loss8�w<


total_loss���>


accuracy_1��?���]       a[��	�4�*_��A�*O

prediction_loss��>

reg_loss,�w<


total_lossPC�>


accuracy_1q=
?�/o]       a[��	�V�*_��A�*O

prediction_loss��?

reg_loss"�w<


total_loss�x?


accuracy_1���>S�{D]       a[��	!v�*_��A�*O

prediction_loss��>

reg_loss�w<


total_lossPC�>


accuracy_1q=
?xmN`]       a[��	ŏ�*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>o��.]       a[��	��*_��A�*O

prediction_loss)\?

reg_loss�w<


total_lossA;?


accuracy_1�G�>C*m]       a[��	�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>����]       a[��	�y�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?^g}	]       a[��	Y�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>�[�?]       a[��	�D�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>�'b]       a[��	Ve�*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossMC�>


accuracy_1q=
?�Q(]       a[��	���*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?����]       a[��	��*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss�>


accuracy_1
�#?V<��]       a[��	]�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>����]       a[��	"S�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?��+]       a[��	�t�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�Ѣq]       a[��	���*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossLC�>


accuracy_1q=
?9Ƽg]       a[��	H��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>��X]       a[��	�߿*_��A�*O

prediction_loss)\?

reg_loss|�w<


total_loss?;?


accuracy_1�G�><�]       a[��	���*_��A�*O

prediction_loss�z?

reg_lossu�w<


total_loss�Y?


accuracy_1=
�>����]       a[��	���*_��A�*O

prediction_loss��(?

reg_lossi�w<


total_loss��,?


accuracy_1{�>��3�]       a[��	��*_��A�*O

prediction_lossq=
?

reg_loss^�w<


total_loss�?


accuracy_1��>��9�]       a[��	w��*_��A�*O

prediction_loss��?

reg_lossT�w<


total_loss�x?


accuracy_1���>K.��]       a[��	��*_��A�*O

prediction_loss��>

reg_lossJ�w<


total_lossIC�>


accuracy_1q=
?�*z0]       a[��	�C�*_��A�*O

prediction_loss�?

reg_loss?�w<


total_loss��?


accuracy_1���>0M@�]       a[��	�g�*_��A�*O

prediction_loss�z?

reg_loss7�w<


total_loss�Y?


accuracy_1=
�>Q�u�]       a[��	��*_��A�*O

prediction_loss�G�>

reg_loss+�w<


total_loss��>


accuracy_1)\?�Sx�]       a[��	д�*_��A�*O

prediction_lossq=
?

reg_loss!�w<


total_loss�?


accuracy_1��>O�r5]       a[��	���*_��A�*O

prediction_loss��?

reg_loss�w<


total_loss�x?


accuracy_1���>�=8�]       a[��	�*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1�?`$]       a[��	�/�*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss�?


accuracy_1   ?�X�+]       a[��	�E�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?%ai�]       a[��	:[�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>���]       a[��	N{�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1��?<��]       a[��	[��*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossFC�>


accuracy_1q=
?��]       a[��	���*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?��l*]       a[��	\��*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�

�]       a[��	��*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?NS-]       a[��	��*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>��UX]       a[��	��*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>ͤ�;]       a[��	�3�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?�>E�]       a[��	�J�*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossCC�>


accuracy_1q=
?9��]       a[��	�_�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?�Yv]       a[��	�|�*_��A�*O

prediction_loss��?

reg_lossz�w<


total_loss�x?


accuracy_1���>���]       a[��	���*_��A�*O

prediction_loss���>

reg_losso�w<


total_loss���>


accuracy_1��?y�h]       a[��	V��*_��A�*O

prediction_loss)\?

reg_losse�w<


total_loss;;?


accuracy_1�G�>]h�]       a[��	|
�*_��A�*O

prediction_loss   ?

reg_lossZ�w<


total_loss�?


accuracy_1   ?�5]       a[��	�1�*_��A�*O

prediction_loss���>

reg_lossP�w<


total_loss���>


accuracy_1�?��h]       a[��	�Z�*_��A�*O

prediction_loss\��>

reg_lossE�w<


total_loss~M�>


accuracy_1R�?Ʈ�]       a[��	��*_��A�*O

prediction_loss
�#?

reg_loss;�w<


total_loss�'?


accuracy_1�Q�>�m�t]       a[��	M��*_��A�*O

prediction_loss
ף>

reg_loss2�w<


total_loss,��>


accuracy_1{.?SK
N]       a[��	R��*_��A�*O

prediction_loss)\?

reg_loss&�w<


total_loss:;?


accuracy_1�G�>��S]       a[��	h��*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?]�]       a[��	r��*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?
���]       a[��	��*_��A�*O

prediction_lossR�?

reg_loss
�w<


total_lossb�"?


accuracy_1\��>v@��]       a[��	��*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>߽�]       a[��		2�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?"�y]       a[��	�I�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?
-h"]       a[��	�`�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?W��]       a[��	||�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�5JW]       a[��	k��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>��A�]       a[��	ͱ�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss=C�>


accuracy_1q=
?�3j]       a[��	��*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss8;?


accuracy_1�G�>�m�m]       a[��	 ��*_��A�*O

prediction_loss
�#?

reg_loss��w<


total_loss�'?


accuracy_1�Q�>�1�]       a[��	���*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossZ��>


accuracy_1�z?���]]       a[��	��*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>B���]       a[��	�3�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>��A�]       a[��	�G�*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss;C�>


accuracy_1q=
?���j]       a[��	 ^�*_��A�*O

prediction_loss=
�>

reg_lossv�w<


total_lossY��>


accuracy_1�z?���]       a[��	�q�*_��A�*O

prediction_loss���>

reg_lossm�w<


total_loss��>


accuracy_1��?~ݐ�]       a[��	"��*_��A�*O

prediction_loss��>

reg_lossd�w<


total_loss:C�>


accuracy_1q=
?����]       a[��	f��*_��A�*O

prediction_lossq=
?

reg_lossZ�w<


total_loss~?


accuracy_1��>���]       a[��	Ժ�*_��A�*O

prediction_loss��?

reg_lossP�w<


total_loss�x?


accuracy_1���>��7S]       a[��	���*_��A�*O

prediction_loss��(?

reg_lossD�w<


total_loss��,?


accuracy_1{�>�+m
]       a[��	���*_��A�*O

prediction_loss��>

reg_loss:�w<


total_loss9C�>


accuracy_1q=
?y�v�]       a[��	.�*_��A�*O

prediction_lossq=
?

reg_loss/�w<


total_loss~?


accuracy_1��>�w?d]       a[��	��*_��A�*O

prediction_lossR�?

reg_loss%�w<


total_loss_�"?


accuracy_1\��>5sJ�]       a[��	�6�*_��A�*O

prediction_loss\��>

reg_loss�w<


total_lossuM�>


accuracy_1R�?}T]       a[��	�R�*_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss5;?


accuracy_1�G�>|Ρ�]       a[��	Ks�*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?[�#�]       a[��	��*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?��u�]       a[��	z��*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss7C�>


accuracy_1q=
?	���]       a[��	j��*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?W �]       a[��	1��*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�]       a[��	-��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss|?


accuracy_1��>��Hb]       a[��	T��*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\??v�]       a[��	��*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>徬U]       a[��	�4�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss5C�>


accuracy_1q=
?P �]       a[��	oJ�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss4C�>


accuracy_1q=
?y��^]       a[��	�a�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1��?$N~]       a[��	k~�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss{?


accuracy_1��>��7~]       a[��	V��*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?��d]       a[��	H��*_��A�*O

prediction_loss�z?

reg_lossx�w<


total_loss�Y?


accuracy_1=
�>�u�]       a[��	��*_��A�*O

prediction_loss��?

reg_lossn�w<


total_loss�x?


accuracy_1���>���<]       a[��	�)�*_��A�*O

prediction_loss�G�>

reg_lossb�w<


total_loss��>


accuracy_1)\?y��]       a[��	�?�*_��A�*O

prediction_loss=
�>

reg_lossY�w<


total_lossP��>


accuracy_1�z?9z�"]       a[��	�X�*_��A�*O

prediction_lossR�?

reg_lossO�w<


total_loss[�"?


accuracy_1\��>�i^]       a[��	To�*_��A�*O

prediction_loss   ?

reg_lossC�w<


total_loss	�?


accuracy_1   ?VA$�]       a[��	Ӈ�*_��A�*O

prediction_loss�?

reg_loss9�w<


total_loss��?


accuracy_1���>��!]       a[��	^��*_��A�*O

prediction_loss�G�>

reg_loss/�w<


total_loss��>


accuracy_1)\? V4�]       a[��	,��*_��A�*O

prediction_loss��>

reg_loss$�w<


total_loss0C�>


accuracy_1q=
?�=��]       a[��	>��*_��A�*O

prediction_loss���>

reg_loss�w<


total_lossފ�>


accuracy_1��?��Ii]       a[��	���*_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Y?


accuracy_1=
�>�9�]       a[��	���*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?�!G\]       a[��	�%�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossy?


accuracy_1��>jyɄ]       a[��	J�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?�d
_]       a[��	ob�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?���z]       a[��	�x�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?;�u]       a[��	C��*_��A�*O

prediction_loss
�#?

reg_loss��w<


total_loss�'?


accuracy_1�Q�>V<��]       a[��		��*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?�e�]       a[��	a��*_��A�*O

prediction_loss{.?

reg_loss��w<


total_loss��1?


accuracy_1
ף>�!¨]       a[��	C��*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>���!]       a[��	��*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss,C�>


accuracy_1q=
?8(�]       a[��	2�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?ҰE�]       a[��	�V�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?���1]       a[��	Ps�*_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossX�"?


accuracy_1\��>B��]       a[��	���*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss+C�>


accuracy_1q=
?��]       a[��	���*_��A�*O

prediction_loss��?

reg_losst�w<


total_loss�x?


accuracy_1���>m�X�]       a[��	R��*_��A�*O

prediction_loss���>

reg_lossj�w<


total_loss���>


accuracy_1�?h���]       a[��	�*_��A�*O

prediction_loss���>

reg_loss`�w<


total_loss؊�>


accuracy_1��?Q��]       a[��	1#�*_��A�*O

prediction_loss   ?

reg_lossT�w<


total_loss�?


accuracy_1   ?%xq5]       a[��	!;�*_��A�*O

prediction_lossq=
?

reg_lossJ�w<


total_lossv?


accuracy_1��>"���]       a[��	TS�*_��A�*O

prediction_loss���>

reg_loss?�w<


total_loss���>


accuracy_1�?��5
]       a[��	�k�*_��A�*O

prediction_loss�z?

reg_loss6�w<


total_loss�Y?


accuracy_1=
�>fa��]       a[��	��*_��A�*O

prediction_loss���>

reg_loss*�w<


total_loss֊�>


accuracy_1��?^���]       a[��	���*_��A�*O

prediction_loss�?

reg_loss �w<


total_loss��?


accuracy_1���>�n~�]       a[��	й�*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_lossu?


accuracy_1��>��F�]       a[��	���*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_lossu?


accuracy_1��>�"�7]       a[��	9��*_��A�*O

prediction_loss{�>

reg_loss �w<


total_loss�ҵ>


accuracy_1��(?3�&]       a[��	��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossu?


accuracy_1��>đ�1]       a[��	+�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?����]       a[��	cF�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossu?


accuracy_1��>H��]       a[��	>]�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss&C�>


accuracy_1q=
?��|]       a[��	ut�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>>�]       a[��	���*_��A�*O

prediction_loss\��>

reg_loss��w<


total_lossbM�>


accuracy_1R�?���]       a[��	à�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss%C�>


accuracy_1q=
?VV��]       a[��	���*_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossU�"?


accuracy_1\��>���O]       a[��	}��*_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossU�"?


accuracy_1\��>��+O]       a[��	���*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?I��]       a[��	��*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>���]       a[��	M�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?
�kF]       a[��	�(�*_��A�*O

prediction_loss�G�>

reg_loss~�w<


total_loss��>


accuracy_1)\?��%]       a[��	�>�*_��A�*O

prediction_loss�z?

reg_lossr�w<


total_loss�Y?


accuracy_1=
�>���]       a[��	&S�*_��A�*O

prediction_loss��(?

reg_lossi�w<


total_loss��,?


accuracy_1{�>��X�]       a[��	g�*_��A�*O

prediction_loss)\?

reg_loss^�w<


total_loss*;?


accuracy_1�G�>R�V]       a[��	�}�*_��A�*O

prediction_loss   ?

reg_lossS�w<


total_loss�?


accuracy_1   ?��0]       a[��	9��*_��A�*O

prediction_loss��>

reg_lossH�w<


total_loss!C�>


accuracy_1q=
?Bj@�]       a[��	��*_��A�*O

prediction_loss���>

reg_loss=�w<


total_lossϊ�>


accuracy_1��?W�k�]       a[��	q��*_��A�*O

prediction_loss��>

reg_loss2�w<


total_loss!C�>


accuracy_1q=
?�n]       a[��	.��*_��A�*O

prediction_loss��>

reg_loss(�w<


total_loss C�>


accuracy_1q=
?�< �]       a[��	m��*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_lossq?


accuracy_1��>p_c]       a[��	��*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1�?BO�]       a[��	�8�*_��A�*O

prediction_loss   ?

reg_loss
�w<


total_loss �?


accuracy_1   ?V���]       a[��	fO�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>��*]       a[��	Vc�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?�A��]       a[��	z�*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>�CL]       a[��	���*_��A�*O

prediction_loss\��>

reg_loss�w<


total_loss[M�>


accuracy_1R�?}r[]       a[��	��*_��A�*O

prediction_loss
�#?

reg_lossֿw<


total_loss	�'?


accuracy_1�Q�>��Vf]       a[��	��*_��A�*O

prediction_loss�G�>

reg_loss̿w<


total_loss��>


accuracy_1)\?Ձ��]       a[��	���*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>��H�]       a[��	B��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossp?


accuracy_1��>'bG]       a[��	��*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss:��>


accuracy_1�z?{�<]       a[��	�!�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?	��1]       a[��	*:�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?_��]       a[��	\Z�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�P��]       a[��	ap�*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossɊ�>


accuracy_1��?�4�]       a[��	���*_��A�*O

prediction_loss�z?

reg_lossy�w<


total_loss�Y?


accuracy_1=
�>%�]       a[��	Н�*_��A�*O

prediction_loss���>

reg_lossm�w<


total_loss���>


accuracy_1�?#��]       a[��	W��*_��A�*O

prediction_loss=
�>

reg_lossc�w<


total_loss8��>


accuracy_1�z?���]       a[��	���*_��A�*O

prediction_loss)\?

reg_lossY�w<


total_loss&;?


accuracy_1�G�>�	gS]       a[��	���*_��A�*O

prediction_loss=
�>

reg_lossP�w<


total_loss8��>


accuracy_1�z?i:�]       a[��	"��*_��A�*O

prediction_loss��>

reg_lossE�w<


total_lossC�>


accuracy_1q=
?̋k�]       a[��	Q�*_��A�*O

prediction_loss\��>

reg_loss<�w<


total_lossVM�>


accuracy_1R�?l�3]       a[��	�.�*_��A�*O

prediction_loss���>

reg_loss0�w<


total_loss���>


accuracy_1�?���)]       a[��	|E�*_��A�*O

prediction_loss)\?

reg_loss&�w<


total_loss&;?


accuracy_1�G�>�ǻ]       a[��	�[�*_��A�*O

prediction_loss��>

reg_loss�w<


total_lossC�>


accuracy_1q=
?�&~�]       a[��	Kr�*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>��@:]       a[��	P��*_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Y?


accuracy_1=
�>���A]       a[��	8��*_��A�*O

prediction_loss
ף>

reg_loss��w<


total_loss��>


accuracy_1{.?��Uo]       a[��	��*_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Y?


accuracy_1=
�>�Q�]       a[��	|��*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1�?�Nm]       a[��	\��*_��A�*O

prediction_loss   ?

reg_lossܾw<


total_loss��?


accuracy_1   ?MԈ�]       a[��	T��*_��A�*O

prediction_lossq=
?

reg_lossѾw<


total_lossl?


accuracy_1��>L���]       a[��	��*_��A�*O

prediction_loss�G�>

reg_lossȾw<


total_loss��>


accuracy_1)\?�vt<]       a[��	x%�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�;g	]       a[��	�D�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>��N]       a[��	`Y�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>s6�]       a[��	qv�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1�?힋]       a[��	���*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossC�>


accuracy_1q=
?�/�]       a[��	��*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossC�>


accuracy_1q=
?���]       a[��	��*_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossL�"?


accuracy_1\��>!�N�]       a[��	���*_��A�*O

prediction_lossq=
?

reg_lossy�w<


total_lossk?


accuracy_1��>��Cr]       a[��	=)�*_��A�*O

prediction_loss�G�>

reg_losse�w<


total_loss��>


accuracy_1)\?}vu]       a[��	�>�*_��A�*O

prediction_loss
�#?

reg_lossZ�w<


total_loss�'?


accuracy_1�Q�>�0��]       a[��	qU�*_��A�*O

prediction_lossq=
?

reg_lossP�w<


total_lossj?


accuracy_1��>n�0�]       a[��	Lo�*_��A�*O

prediction_loss�G�>

reg_lossF�w<


total_loss��>


accuracy_1)\?F���]       a[��	j��*_��A�*O

prediction_loss   ?

reg_loss;�w<


total_loss��?


accuracy_1   ?Ҍ�b]       a[��	���*_��A�*O

prediction_loss)\?

reg_loss0�w<


total_loss";?


accuracy_1�G�>E|B&]       a[��	#��*_��A�*O

prediction_loss���>

reg_loss$�w<


total_loss���>


accuracy_1�?uH^r]       a[��	���*_��A�*O

prediction_loss{.?

reg_loss�w<


total_losss�1?


accuracy_1
ף>��]       a[��	���*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_lossi?


accuracy_1��>$�]       a[��	u�*_��A�*O

prediction_loss
�#?

reg_loss�w<


total_loss�'?


accuracy_1�Q�>`��]       a[��	L�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss!;?


accuracy_1�G�>ft4w]       a[��	D1�*_��A�*O

prediction_loss{.?

reg_loss�w<


total_losss�1?


accuracy_1
ף>,xz�]       a[��	�H�*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_lossi?


accuracy_1��>%Q�>]       a[��	(b�*_��A�*O

prediction_loss   ?

reg_loss޽w<


total_loss��?


accuracy_1   ?_��]       a[��	�*_��A�*O

prediction_loss���>

reg_lossԽw<


total_loss���>


accuracy_1��?R;�]       a[��	���*_��A�*O

prediction_loss���>

reg_lossɽw<


total_loss���>


accuracy_1��?�q��]       a[��	H��*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?� s�]       a[��	���*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss}��>


accuracy_1�?W�]       a[��	� �*_��A�*O

prediction_loss\��>

reg_loss��w<


total_lossIM�>


accuracy_1R�?���]       a[��	+j�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss ;?


accuracy_1�G�>jd��]       a[��	"��*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss|��>


accuracy_1�?�,�C]       a[��	���*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss;?


accuracy_1�G�>���]       a[��	�	�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�&]       a[��	1��*_��A�*O

prediction_loss   ?

reg_lossz�w<


total_loss��?


accuracy_1   ?[�{]       a[��	���*_��A�*O

prediction_loss���>

reg_losso�w<


total_lossz��>


accuracy_1�?��:]       a[��	��*_��A�*O

prediction_loss�?

reg_lossd�w<


total_loss��?


accuracy_1���>Yƃ]       a[��	�h�*_��A�*O

prediction_loss)\?

reg_lossY�w<


total_loss;?


accuracy_1�G�>�[4�]       a[��	d��*_��A�*O

prediction_loss�z?

reg_lossP�w<


total_loss�Y?


accuracy_1=
�>dT�]       a[��	��*_��A�*O

prediction_loss   ?

reg_lossE�w<


total_loss��?


accuracy_1   ?�v�
]       a[��	���*_��A�*O

prediction_lossq=
?

reg_loss;�w<


total_lossf?


accuracy_1��>����]       a[��	
�*_��A�*O

prediction_loss   ?

reg_loss/�w<


total_loss��?


accuracy_1   ?6U�#]       a[��	QK�*_��A�*O

prediction_loss��>

reg_loss%�w<


total_lossC�>


accuracy_1q=
?4	: ]       a[��	Ww�*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_losse?


accuracy_1��>�]��]       a[��	B��*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_losse?


accuracy_1��>�[ �]       a[��	I��*_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss%��>


accuracy_1�z?id��]       a[��	��*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossw��>


accuracy_1�?J��]       a[��	�H�*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>��#]       a[��	7l�*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?@5�]       a[��	s��*_��A�*O

prediction_loss   ?

reg_lossݼw<


total_loss��?


accuracy_1   ?��Q�]       a[��	g��*_��A�*O

prediction_loss)\?

reg_lossҼw<


total_loss;?


accuracy_1�G�>|�m]       a[��	�(�*_��A�*O

prediction_loss   ?

reg_lossɼw<


total_loss��?


accuracy_1   ?��^\]       a[��		P�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?C~}]       a[��	Fx�*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossu��>


accuracy_1�?$�Q]       a[��	���*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossC�>


accuracy_1q=
?3���]       a[��	���*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?h���]       a[��	�.�*_��A�*O

prediction_loss\��>

reg_loss��w<


total_lossAM�>


accuracy_1R�?�Ԇ�]       a[��	�Z�*_��A�*O

prediction_loss��>

reg_loss��w<


total_lossC�>


accuracy_1q=
?���]       a[��	�*_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss@M�>


accuracy_1R�?���]       a[��	���*_��A�*O

prediction_loss   ?

reg_losst�w<


total_loss��?


accuracy_1   ?s=��]       a[��	j��*_��A�*O

prediction_loss�?

reg_lossj�w<


total_loss��?


accuracy_1���>pH��]       a[��	�H�*_��A�*O

prediction_loss�z?

reg_loss_�w<


total_loss�Y?


accuracy_1=
�>�d:�]       a[��	�s�*_��A�*O

prediction_loss��?

reg_lossU�w<


total_loss�x?


accuracy_1���>��7�]       a[��	D��*_��A�*O

prediction_loss�G�>

reg_lossI�w<


total_loss��>


accuracy_1)\?!�-]       a[��	���*_��A�*O

prediction_loss���>

reg_loss>�w<


total_loss���>


accuracy_1��?A�[�]       a[��	J'�*_��A�*O

prediction_loss�Q�>

reg_loss4�w<


total_loss��>


accuracy_1
�#?{��]       a[��	�T�*_��A�*O

prediction_loss=
�>

reg_loss+�w<


total_loss��>


accuracy_1�z?���]       a[��	���*_��A�*O

prediction_loss�Q8?

reg_loss�w<


total_loss�0<?


accuracy_1)\�>aF�W]       a[��	���*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_lossa?


accuracy_1��>ލ8[]       a[��	��*_��A�*O

prediction_loss�?

reg_loss
�w<


total_loss��?


accuracy_1���>H�>�]       a[��	N��*_��A�*O

prediction_lossR�?

reg_loss �w<


total_lossB�"?


accuracy_1\��>�,(]       a[��	:�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�
]       a[��	�s�*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>S}8]       a[��	��*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?�)�-]       a[��	&��*_��A�*O

prediction_loss���>

reg_loss׻w<


total_lossn��>


accuracy_1�?�cł]       a[��	��*_��A�*O

prediction_loss\��>

reg_lossλw<


total_loss:M�>


accuracy_1R�?m|[�]       a[��	�@�*_��A�*O

prediction_loss
ף>

reg_lossûw<


total_loss蔫>


accuracy_1{.?��5�]       a[��	j�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss;?


accuracy_1�G�>S�ƶ]       a[��	~��*_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�x?


accuracy_1���>O*9]       a[��	г�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�B�>


accuracy_1q=
?۪L�]       a[��	X��*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossl��>


accuracy_1�?Z�]       a[��	'1�*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossl��>


accuracy_1�?�!�`]       a[��	U�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�B�>


accuracy_1q=
?��=�]       a[��	lx�*_��A�*O

prediction_loss��(?

reg_loss{�w<


total_loss��,?


accuracy_1{�>PH]       a[��	���*_��A�*O

prediction_loss�G�>

reg_lossp�w<


total_loss��>


accuracy_1)\?�*R]       a[��	���*_��A�*O

prediction_loss�?

reg_lossf�w<


total_loss��?


accuracy_1���>>��]       a[��	q��*_��A�*O

prediction_lossq=
?

reg_lossZ�w<


total_loss^?


accuracy_1��>]�]       a[��	��*_��A�*O

prediction_loss   ?

reg_lossP�w<


total_loss��?


accuracy_1   ?�]       a[��	�2�*_��A�*O

prediction_loss�z?

reg_lossE�w<


total_loss�Y?


accuracy_1=
�>ǚɱ]       a[��	�O�*_��A�*O

prediction_loss   ?

reg_loss=�w<


total_loss��?


accuracy_1   ?q��X]       a[��	<i�*_��A�*O

prediction_loss���>

reg_loss3�w<


total_loss���>


accuracy_1��?5Av�]       a[��	�}�*_��A�*O

prediction_loss)\?

reg_loss)�w<


total_loss;?


accuracy_1�G�>���]]       a[��	2��*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?�VT�]       a[��	q��*_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Y?


accuracy_1=
�>}"n]       a[��	}��*_��A�*O

prediction_loss   ?

reg_loss
�w<


total_loss��?


accuracy_1   ?n&�]       a[��	���*_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Y?


accuracy_1=
�>����]       a[��	X �*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?��]       a[��	�*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?�'՜]       a[��	�*�*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�B�>


accuracy_1q=
?9��]       a[��	E�*_��A�*O

prediction_loss�z?

reg_lossۺw<


total_loss�Y?


accuracy_1=
�>+��]       a[��	 c�*_��A�*O

prediction_loss�?

reg_lossкw<


total_loss��?


accuracy_1���>C?w]       a[��	�x�*_��A�*O

prediction_loss=
�>

reg_lossĺw<


total_loss��>


accuracy_1�z?'7�\]       a[��	e��*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss;?


accuracy_1�G�>T��]       a[��	7��*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss;?


accuracy_1�G�>�d=]       a[��	¾�*_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss=�"?


accuracy_1\��>=m:*]       a[��	���*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss[?


accuracy_1��>L!��]       a[��	���*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss;?


accuracy_1�G�>d���]       a[��	N�*_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss0M�>


accuracy_1R�?����]       a[��	��*_��A�*O

prediction_loss�?

reg_loss{�w<


total_loss��?


accuracy_1���>&���]       a[��	�0�*_��A�*O

prediction_loss�G�>

reg_losso�w<


total_loss��>


accuracy_1)\?`P]       a[��	�F�*_��A�*O

prediction_loss)\?

reg_lossg�w<


total_loss;?


accuracy_1�G�>�u�]       a[��	���*_��A�*O

prediction_loss�z?

reg_lossS�w<


total_loss�Y?


accuracy_1=
�>]	�]       a[��	7��*_��A�*O

prediction_loss���>

reg_lossG�w<


total_loss���>


accuracy_1��?]�8]       a[��		��*_��A�*O

prediction_loss��>

reg_loss>�w<


total_loss�B�>


accuracy_1q=
?ݨ�]       a[��	���*_��A�*O

prediction_loss
ף>

reg_loss4�w<


total_lossܔ�>


accuracy_1{.?��rW]       a[��	���*_��A�*O

prediction_loss   ?

reg_loss*�w<


total_loss��?


accuracy_1   ?H�-]       a[��	�	�*_��A�*O

prediction_lossR�?

reg_loss �w<


total_loss:�"?


accuracy_1\��>V[3]       a[��	�!�*_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss;?


accuracy_1�G�>IZ9�]       a[��	�2�*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�B�>


accuracy_1q=
?��]       a[��	�C�*_��A�*O

prediction_lossq=
?

reg_loss �w<


total_lossY?


accuracy_1��>�� �]       a[��	�\�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?��^�]       a[��	�v�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�ҁ�]       a[��	~��*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�B�>


accuracy_1q=
??�u�]       a[��	��*_��A�*O

prediction_loss�z?

reg_lossٹw<


total_loss�Y?


accuracy_1=
�>u_B<]       a[��	��*_��A�*O

prediction_loss   ?

reg_lossιw<


total_loss��?


accuracy_1   ?Nl�]       a[��	J��*_��A�*O

prediction_loss�G�>

reg_lossùw<


total_loss|�>


accuracy_1)\?ߘ��]       a[��	G��*_��A�*O

prediction_loss{�>

reg_loss��w<


total_lossIҵ>


accuracy_1��(?_��v]       a[��	n��*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?܁Q]       a[��	#�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossX?


accuracy_1��>)r�]       a[��	�*�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?���%]       a[��	�@�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�Cbe]       a[��	zV�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss[��>


accuracy_1�?���]       a[��	�o�*_��A�*O

prediction_loss�G�>

reg_lossz�w<


total_lossz�>


accuracy_1)\?��/�]       a[��	*��*_��A�*O

prediction_lossq=
?

reg_lossp�w<


total_lossW?


accuracy_1��>�z]�]       a[��	���*_��A�*O

prediction_loss���>

reg_lossh�w<


total_lossZ��>


accuracy_1�?揬g]       a[��	!��*_��A�*O

prediction_loss��>

reg_loss]�w<


total_loss�B�>


accuracy_1q=
?Li��]       a[��	���*_��A�*O

prediction_loss)\?

reg_lossR�w<


total_loss;?


accuracy_1�G�>����]       a[��	��*_��A�*O

prediction_loss=
�>

reg_lossI�w<


total_loss��>


accuracy_1�z?k�'�]       a[��	�]�*_��A�*O

prediction_loss�?

reg_loss?�w<


total_loss��?


accuracy_1���>�!w�]       a[��	��*_��A�*O

prediction_loss��>

reg_loss6�w<


total_loss�B�>


accuracy_1q=
?][��]       a[��	���*_��A�*O

prediction_lossR�?

reg_loss+�w<


total_loss7�"?


accuracy_1\��>��#�]       a[��	t��*_��A�*O

prediction_loss�Q�>

reg_loss�w<


total_loss��>


accuracy_1
�#?Ol�]       a[��	O��*_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss��>


accuracy_1�z?e3�B]       a[��	���*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�B�>


accuracy_1q=
?���]       a[��	���*_��A�*O

prediction_loss���>

reg_loss�w<


total_lossW��>


accuracy_1�?\.(~]       a[��	M�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss;?


accuracy_1�G�>�ᑧ]       a[��	�,�*_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss��>


accuracy_1�z?uY�]       a[��	�C�*_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>��t]       a[��	�[�*_��A�*O

prediction_loss��>

reg_lossٸw<


total_loss�B�>


accuracy_1q=
?pk�F]       a[��	�s�*_��A�*O

prediction_lossq=
?

reg_loss̸w<


total_lossT?


accuracy_1��>�{�]       a[��	ڌ�*_��A�*O

prediction_loss   ?

reg_lossøw<


total_loss��?


accuracy_1   ?�y]       a[��	���*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_losst�>


accuracy_1)\?lqC]       a[��	���*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?��-�]       a[��	
��*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�B�>


accuracy_1q=
?�j]       a[��	���*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>��]       a[��	�*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss��>


accuracy_1�z?�	�}]       a[��	K �*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�f70]       a[��	�6�*_��A�*O

prediction_loss   ?

reg_lossy�w<


total_loss��?


accuracy_1   ?�K&]       a[��	IM�*_��A�*O

prediction_loss�?

reg_lossm�w<


total_loss��?


accuracy_1���>��V]       a[��	Ac�*_��A�*O

prediction_loss���>

reg_lossb�w<


total_lossR��>


accuracy_1�?T�]       a[��	�z�*_��A�*O

prediction_loss���>

reg_lossY�w<


total_loss���>


accuracy_1��?�i9"]       a[��	G��*_��A�*O

prediction_loss   ?

reg_lossM�w<


total_loss��?


accuracy_1   ?�8]       a[��	���*_��A�*O

prediction_loss���>

reg_lossB�w<


total_loss���>


accuracy_1��?����]       a[��	��*_��A�*O

prediction_loss   ?

reg_loss6�w<


total_loss��?


accuracy_1   ?(��7]       a[��	���*_��A�*O

prediction_loss�G�>

reg_loss-�w<


total_losso�>


accuracy_1)\?���!]       a[��	���*_��A�*O

prediction_loss�G�>

reg_loss$�w<


total_losso�>


accuracy_1)\?���]       a[��	G�*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�B�>


accuracy_1q=
?Y��P]       a[��	��*_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Y?


accuracy_1=
�>��@"]       a[��	�0�*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1��?=Yq�]       a[��	�E�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>#3��]       a[��	�\�*_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss	;?


accuracy_1�G�>\bڝ]       a[��	Kv�*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�B�>


accuracy_1q=
?�n��]       a[��	���*_��A�*O

prediction_loss���>

reg_lossܷw<


total_lossN��>


accuracy_1�?3iC:]       a[��	j��*_��A�*O

prediction_loss��>

reg_lossҷw<


total_loss�B�>


accuracy_1q=
?>�;^]       a[��	'��*_��A�*O

prediction_loss=
�>

reg_lossƷw<


total_loss���>


accuracy_1�z?\0�]       a[��	0��*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossl�>


accuracy_1)\?�u]       a[��	��*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>E�}]       a[��	u�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossP?


accuracy_1��>���c]       a[��	��*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?��1]       a[��	5�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossO?


accuracy_1��>�3]       a[��	�M�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossO?


accuracy_1��>�G��]       a[��	�d�*_��A�*O

prediction_loss��>

reg_loss}�w<


total_loss�B�>


accuracy_1q=
?�d,']       a[��	�x�*_��A�*O

prediction_loss���>

reg_losss�w<


total_loss���>


accuracy_1��?f��]       a[��	є�*_��A�*O

prediction_lossq=
?

reg_lossi�w<


total_lossO?


accuracy_1��>�+�K]       a[��	6��*_��A�*O

prediction_loss���>

reg_loss^�w<


total_lossJ��>


accuracy_1�?z��Y]       a[��	���*_��A�*O

prediction_loss   ?

reg_lossU�w<


total_loss��?


accuracy_1   ?۱{�]       a[��	���*_��A�*O

prediction_loss���>

reg_lossI�w<


total_lossTW�>


accuracy_1333?]q~]       a[��	��*_��A�*O

prediction_loss���>

reg_loss?�w<


total_loss���>


accuracy_1��?��(�]       a[��	f�*_��A�*O

prediction_loss�z?

reg_loss5�w<


total_loss�Y?


accuracy_1=
�>x���]       a[��	(,�*_��A�*O

prediction_loss=
�>

reg_loss+�w<


total_loss���>


accuracy_1�z?`p%]       a[��	B�*_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?u�1�]       a[��	�X�*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_lossg�>


accuracy_1)\?��j]       a[��	rn�*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�B�>


accuracy_1q=
?"�;M]       a[��	���*_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss;?


accuracy_1�G�>�+�]       a[��	L��*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>���]       a[��	{��*_��A�*O

prediction_loss���>

reg_loss�w<


total_lossF��>


accuracy_1�?FM4/]       a[��	���*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�B�>


accuracy_1q=
?�ݪ�]       a[��	���*_��A�*O

prediction_loss���>

reg_lossضw<


total_lossF��>


accuracy_1�?�w��]       a[��	��*_��A�*O

prediction_loss��?

reg_lossζw<


total_lossux?


accuracy_1���>,�T]       a[��	H�*_��A�*O

prediction_loss���>

reg_lossĶw<


total_lossE��>


accuracy_1�?9�]       a[��	�+�*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>�C�9]       a[��	�C�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?F�R=]       a[��	h[�*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossL?


accuracy_1��>"�m]       a[��	�r�*_��A�*O

prediction_loss
�#?

reg_loss��w<


total_loss�'?


accuracy_1�Q�>ph��]       a[��	7��*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossK?


accuracy_1��>��E]       a[��	Ϡ�*_��A�*O

prediction_loss\��>

reg_loss��w<


total_lossM�>


accuracy_1R�?�B�]       a[��	=��*_��A�*O

prediction_loss   ?

reg_lossz�w<


total_loss��?


accuracy_1   ?5b�n]       a[��	���*_��A�*O

prediction_loss=
�>

reg_lossp�w<


total_loss���>


accuracy_1�z?��7�]       a[��	��*_��A�*O

prediction_loss��>

reg_lossg�w<


total_loss�B�>


accuracy_1q=
?؍+�]       a[��	���*_��A�*O

prediction_loss=
�>

reg_loss\�w<


total_loss���>


accuracy_1�z?�o�]       a[��	��*_��A�*O

prediction_loss�G�>

reg_lossS�w<


total_lossa�>


accuracy_1)\?g��
]       a[��	�f�*_��A�*O

prediction_loss�z?

reg_loss=�w<


total_loss�Y?


accuracy_1=
�>2��]       a[��	�|�*_��A�*O

prediction_loss��?

reg_loss2�w<


total_losssx?


accuracy_1���>f�9]       a[��	Ԟ�*_��A�*O

prediction_loss�?

reg_loss)�w<


total_loss��?


accuracy_1���>Q��]       a[��	���*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_lossI?


accuracy_1��>�(2�]       a[��	3��*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss@��>


accuracy_1�?J�M]       a[��	���*_��A�*O

prediction_loss�?

reg_loss
�w<


total_loss��?


accuracy_1���>wE��]       a[��	�%�*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss?��>


accuracy_1�?��]       a[��	�E�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss}��>


accuracy_1��?*2G]       a[��	�b�*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss;?


accuracy_1�G�>�͚�]       a[��	��*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss|��>


accuracy_1��?M\�g]       a[��	���*_��A�*O

prediction_loss   ?

reg_lossٵw<


total_loss��?


accuracy_1   ?uy&]       a[��	���*_��A�*O

prediction_loss���>

reg_loss͵w<


total_loss=��>


accuracy_1�?1�g]       a[��	��*_��A�*O

prediction_loss���>

reg_lossõw<


total_loss=��>


accuracy_1�?_���]       a[��	_@�*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?~�˰]       a[��	�l�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss\�>


accuracy_1)\?	R�]       a[��	p��*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss[�>


accuracy_1)\? #�]       a[��	�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�m��]       a[��	�W�*_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss(�"?


accuracy_1\��>d��0]       a[��	5��*_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?ׯ]       a[��	��*_��A�*O

prediction_loss�z?

reg_lossy�w<


total_loss�Y?


accuracy_1=
�>@��]       a[��	�*_��A�*O

prediction_loss=
�>

reg_losso�w<


total_loss���>


accuracy_1�z?і]       a[��	�O�*_��A�*O

prediction_loss��>

reg_lossd�w<


total_loss�B�>


accuracy_1q=
?y��]       a[��	��*_��A�*O

prediction_loss�z?

reg_loss[�w<


total_loss�Y?


accuracy_1=
�>�W��]       a[��	��*_��A�*O

prediction_loss���>

reg_lossP�w<


total_loss:��>


accuracy_1�?�5&]       a[��	�?�*_��A�*O

prediction_loss\��>

reg_lossE�w<


total_lossM�>


accuracy_1R�?�9,�]       a[��	�`�*_��A�*O

prediction_loss   ?

reg_loss=�w<


total_loss��?


accuracy_1   ?����]       a[��	:��*_��A�*O

prediction_loss��?

reg_loss2�w<


total_lossox?


accuracy_1���>U1��]       a[��	"�*_��A�*O

prediction_loss�?

reg_loss(�w<


total_loss��?


accuracy_1���> Z�]       a[��	�H�*_��A�*O

prediction_loss���>

reg_loss�w<


total_lossCW�>


accuracy_1333?�Mo]       a[��	�r�*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?z�vm]       a[��	��*_��A�*O

prediction_loss��>

reg_loss	�w<


total_loss�B�>


accuracy_1q=
?��FJ]       a[��	���*_��A�*O

prediction_loss
ף>

reg_loss �w<


total_loss���>


accuracy_1{.?=���]       a[��	k(�*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossu��>


accuracy_1��?$��]       a[��	�V�*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�B�>


accuracy_1q=
?t�c1]       a[��	s�*_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?�Z�]       a[��	8��*_��A�*O

prediction_loss=
�>

reg_lossִw<


total_loss���>


accuracy_1�z?�	]       a[��	��*_��A�*O

prediction_lossq=
?

reg_lossʹw<


total_lossD?


accuracy_1��>�Ĵ�]       a[��	���*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>���]       a[��	�N�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>#���]       a[��	���*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?���i]       a[��	t��*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>�ĵ�]       a[��	���*_��A�*O

prediction_lossq=
?

reg_loss��w<


total_lossC?


accuracy_1��>�o]       a[��	���*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�oR ]       a[��	:#�*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossq��>


accuracy_1��?:@��]       a[��	�m�*_��A�*O

prediction_loss
ף>

reg_lossy�w<


total_loss���>


accuracy_1{.?��e]       a[��	ԙ�*_��A�*O

prediction_loss�z?

reg_lossp�w<


total_loss�Y?


accuracy_1=
�>���]       a[��	U��*_��A�*O

prediction_loss�?

reg_lossf�w<


total_loss��?


accuracy_1���>K/?�]       a[��	���*_��A�*O

prediction_loss��>

reg_loss\�w<


total_loss�B�>


accuracy_1q=
?<�S|]       a[��	a�*_��A�*O

prediction_loss���>

reg_lossO�w<


total_loss1��>


accuracy_1�?�XL]       a[��	�s�*_��A�*O

prediction_loss��>

reg_lossG�w<


total_loss�B�>


accuracy_1q=
?<,�]       a[��	Ҧ�*_��A�*O

prediction_lossR�?

reg_loss>�w<


total_loss#�"?


accuracy_1\��>�0�]       a[��	S��*_��A�*O

prediction_loss=
�>

reg_loss1�w<


total_loss���>


accuracy_1�z?��#�]       a[��	���*_��A�*O

prediction_loss��>

reg_loss(�w<


total_loss�B�>


accuracy_1q=
?\�]       a[��	<�*_��A�*O

prediction_lossq=
?

reg_loss�w<


total_lossA?


accuracy_1��>Q���]       a[��	nR�*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_lossO�>


accuracy_1)\?P^��]       a[��	��*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss/��>


accuracy_1�?�
�_]       a[��	���*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ? �@�]       a[��	���*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>�E�]       a[��	
�*_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Y?


accuracy_1=
�>�FMY]       a[��	MI�*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss.��>


accuracy_1�?�:"]       a[��	�n�*_��A�*O

prediction_loss���>

reg_lossճw<


total_lossl��>


accuracy_1��?6 '�]       a[��	���*_��A�*O

prediction_loss�?

reg_loss˳w<


total_loss��?


accuracy_1���>��?�]       a[��	��*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss-��>


accuracy_1�?��S]       a[��	���*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss-��>


accuracy_1�?�_�(]       a[��	�#�*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?���]       a[��	�N�*_��A�*O

prediction_loss���>

reg_loss��w<


total_loss,��>


accuracy_1�?�h8�]       a[��	�z�*_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�B�>


accuracy_1q=
?�fY]       a[��	v��*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?����]       a[��	_��*_��A�*O

prediction_loss���>

reg_loss��w<


total_lossi��>


accuracy_1��?NVZ]       a[��	�A�*_��A�*O

prediction_loss)\?

reg_lossy�w<


total_loss�:?


accuracy_1�G�>�;Z�]       a[��	�d�*_��A�*O

prediction_loss)\?

reg_losso�w<


total_loss�:?


accuracy_1�G�>�fu�]       a[��	���*_��A�*O

prediction_loss)\?

reg_lossd�w<


total_loss�:?


accuracy_1�G�>�4>T]       a[��	$��*_��A�*O

prediction_loss=
�>

reg_lossZ�w<


total_loss���>


accuracy_1�z?Lp��]       a[��	M��*_��A�*O

prediction_loss�G�>

reg_lossO�w<


total_lossH�>


accuracy_1)\?E��F]       a[��	'1�*_��A�*O

prediction_loss��>

reg_lossD�w<


total_loss�B�>


accuracy_1q=
?�ف�]       a[��	�]�*_��A�*O

prediction_lossR�?

reg_loss8�w<


total_loss�"?


accuracy_1\��>3bg�]       a[��	��*_��A�*O

prediction_lossq=
?

reg_loss0�w<


total_loss>?


accuracy_1��>e(]       a[��	���*_��A�*O

prediction_loss���>

reg_loss%�w<


total_lossf��>


accuracy_1��?ͦ�,]       a[��	I��*_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�B�>


accuracy_1q=
?�5�t]       a[��	��*_��A�*O

prediction_loss��?

reg_loss�w<


total_lossfx?


accuracy_1���>�H�]       a[��	�7�*_��A�*O

prediction_loss���>

reg_loss�w<


total_loss'��>


accuracy_1�?��5]       a[��	�U�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossF�>


accuracy_1)\?��]       a[��	u�*_��A�*O

prediction_loss�G�>

reg_loss�w<


total_lossF�>


accuracy_1)\?e@��]       a[��	؛�*_��A�*O

prediction_loss��?

reg_loss�w<


total_lossfx?


accuracy_1���>Cn+�]       a[��	^��*_��A�*O

prediction_loss��>

reg_loss߲w<


total_loss�B�>


accuracy_1q=
?U=:�]       a[��	���*_��A�*O

prediction_lossq=
?

reg_lossԲw<


total_loss<?


accuracy_1��>�i]       a[��	��*_��A�*O

prediction_loss   ?

reg_loss˲w<


total_loss��?


accuracy_1   ?�7�%]       a[��	�)�*_��A�*O

prediction_loss��?

reg_loss��w<


total_lossex?


accuracy_1���>P�#%]       a[��	Q�*_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>2���]       a[��	Qk�*_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossC�>


accuracy_1)\?��9]       a[��	���*_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�dC�]       a[��	'��*_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>/d��]       a[��	���*_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?��=0]       a[��	���*_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>�F�m]       a[��	
�*_��A�*O

prediction_loss���>

reg_lossv�w<


total_loss#��>


accuracy_1�?��Ȧ]       a[��	&4�*_��A�*O

prediction_loss��>

reg_lossn�w<


total_loss�B�>


accuracy_1q=
?x�R']       a[��	�V�*_��A�*O

prediction_loss��>

reg_lossb�w<


total_loss�B�>


accuracy_1q=
?�4j]       a[��	l|�*_��A�*O

prediction_loss�z?

reg_lossW�w<


total_loss�Y?


accuracy_1=
�>�SoD]       a[��	I��*_��A�*O

prediction_loss�?

reg_lossN�w<


total_loss��?


accuracy_1���>s��]       a[��	ؼ�*_��A�*O

prediction_loss�z?

reg_lossD�w<


total_loss�Y?


accuracy_1=
�>��'F]       a[��	* +_��A�*O

prediction_loss���>

reg_loss/�w<


total_loss ��>


accuracy_1�?:0�]       a[��	B +_��A�*O

prediction_lossq=
?

reg_loss$�w<


total_loss:?


accuracy_1��>���_]       a[��	a +_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?��]       a[��	I� +_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss9?


accuracy_1��>���0]       a[��	� +_��A�*O

prediction_lossR�?

reg_loss�w<


total_loss�"?


accuracy_1\��>q�N]       a[��	�� +_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss9?


accuracy_1��>�� �]       a[��	]� +_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss9?


accuracy_1��>wC�]       a[��	�� +_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Y?


accuracy_1=
�>v���]       a[��	v+_��A�*O

prediction_loss�Q�>

reg_lossݱw<


total_loss{�>


accuracy_1
�#?�ZS�]       a[��	�:+_��A�*O

prediction_loss{�>

reg_lossѱw<


total_loss
ҵ>


accuracy_1��(?N�A�]       a[��	X+_��A�*O

prediction_loss�G�>

reg_lossƱw<


total_loss<�>


accuracy_1)\?���]       a[��	Wy+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�B�>


accuracy_1q=
?��7�]       a[��	��+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>?�]       a[��	U�+_��A�*O

prediction_loss
�#?

reg_loss��w<


total_lossѵ'?


accuracy_1�Q�>荅�]       a[��	��+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss~�?


accuracy_1���>o(�K]       a[��	��+_��A�*O

prediction_loss{.?

reg_loss��w<


total_lossA�1?


accuracy_1
ף>%�A�]       a[��	T+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>�
�]       a[��	�P+_��A�*O

prediction_loss�?

reg_loss~�w<


total_loss~�?


accuracy_1���>��L]       a[��	{+_��A�*O

prediction_loss
ף>

reg_losss�w<


total_loss���>


accuracy_1{.?7[aF]       a[��	��+_��A�*O

prediction_loss=
�>

reg_lossi�w<


total_loss���>


accuracy_1�z?]�k�]       a[��	d�+_��A�*O

prediction_loss�?

reg_loss^�w<


total_loss}�?


accuracy_1���>p6?�]       a[��	(
+_��A�*O

prediction_loss��>

reg_lossT�w<


total_loss�B�>


accuracy_1q=
?��J�]       a[��	�/+_��A�*O

prediction_loss   ?

reg_lossK�w<


total_loss��?


accuracy_1   ?x�)]       a[��	�J+_��A�*O

prediction_loss)\?

reg_loss@�w<


total_loss�:?


accuracy_1�G�>T�]       a[��	m+_��A�*O

prediction_loss)\?

reg_loss6�w<


total_loss�:?


accuracy_1�G�>��h3]       a[��	��+_��A�*O

prediction_loss��>

reg_loss.�w<


total_loss�B�>


accuracy_1q=
?Y�FW]       a[��	g�+_��A�*O

prediction_loss��>

reg_loss"�w<


total_loss�B�>


accuracy_1q=
?E��]       a[��	��+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?�Ⳏ]       a[��	�+_��A�*O

prediction_loss��(?

reg_loss�w<


total_loss��,?


accuracy_1{�>��f]       a[��	!+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?C7�]       a[��	�B+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss|�?


accuracy_1���>��`]       a[��	jg+_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Y?


accuracy_1=
�>\[��]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?'��]       a[��	a�+_��A�*O

prediction_loss���>

reg_lossذw<


total_loss��>


accuracy_1�?�j�]       a[��	�+_��A�*O

prediction_loss�G�>

reg_lossϰw<


total_loss4�>


accuracy_1)\?q3�]       a[��	��+_��A�*O

prediction_lossq=
?

reg_lossŰw<


total_loss4?


accuracy_1��>�W�]       a[��	��+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss4?


accuracy_1��>&�O�]       a[��	�+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>f��o]       a[��	�*+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�B�>


accuracy_1q=
?"ޛ�]       a[��	N+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>�%8]       a[��	�m+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>�e<�]       a[��	��+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss3?


accuracy_1��>�Xmd]       a[��	��+_��A�*O

prediction_lossq=
?

reg_lossz�w<


total_loss3?


accuracy_1��>E3]       a[��	��+_��A�*O

prediction_loss���>

reg_lossq�w<


total_loss��>


accuracy_1�?:P��]       a[��	��+_��A�*O

prediction_loss   ?

reg_lossf�w<


total_loss��?


accuracy_1   ?Jؑ]       a[��	�+_��A�*O

prediction_loss�G�>

reg_loss]�w<


total_loss1�>


accuracy_1)\?9q�]       a[��	�*+_��A�*O

prediction_loss   ?

reg_lossS�w<


total_loss��?


accuracy_1   ?���]       a[��	K+_��A�*O

prediction_loss���>

reg_lossJ�w<


total_loss��>


accuracy_1�?����]       a[��	�m+_��A�*O

prediction_loss�G�>

reg_loss@�w<


total_loss0�>


accuracy_1)\?�t f]       a[��	6�+_��A�*O

prediction_loss)\?

reg_loss4�w<


total_loss�:?


accuracy_1�G�>�Q�]       a[��	ҫ+_��A�*O

prediction_loss\��>

reg_loss)�w<


total_loss�L�>


accuracy_1R�?z�އ]       a[��	P�+_��A�*O

prediction_loss��?

reg_loss�w<


total_lossZx?


accuracy_1���>{`�<]       a[��	��+_��A�*O

prediction_loss�?

reg_loss�w<


total_lossx�?


accuracy_1���>�,��]       a[��	��+_��A�*O

prediction_loss)\?

reg_loss	�w<


total_loss�:?


accuracy_1�G�>}VQ]       a[��	+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?��
R]       a[��	5+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�ڤ*]       a[��	�T+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�B�>


accuracy_1q=
?�g�]       a[��	�q+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?��]       a[��	�+_��A�*O

prediction_loss�G�>

reg_loss֯w<


total_loss-�>


accuracy_1)\?>Y7]       a[��	�+_��A�*O

prediction_loss   ?

reg_loss˯w<


total_loss��?


accuracy_1   ?u`�]       a[��	�+_��A�*O

prediction_loss)\?

reg_loss¯w<


total_loss�:?


accuracy_1�G�>͕'�]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?��R�]       a[��	a+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?8˰1]       a[��	�5+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�B�>


accuracy_1q=
?b�\�]       a[��	�Q+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>ᴽ�]       a[��	�l+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�B�>


accuracy_1q=
?����]       a[��	��+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>�2��]       a[��	��+_��A�*O

prediction_loss�G�>

reg_lossz�w<


total_loss*�>


accuracy_1)\?X�۠]       a[��	��+_��A�*O

prediction_loss�z?

reg_lossq�w<


total_loss�Y?


accuracy_1=
�>Xe�i]       a[��	��+_��A�*O

prediction_lossq=
?

reg_lossd�w<


total_loss/?


accuracy_1��>�%�U]       a[��	�	+_��A�*O

prediction_loss�?

reg_loss\�w<


total_lossu�?


accuracy_1���>k��]       a[��	�/	+_��A�*O

prediction_loss=
�>

reg_lossO�w<


total_loss���>


accuracy_1�z?��G]       a[��	J	+_��A�*O

prediction_loss�?

reg_lossF�w<


total_lossu�?


accuracy_1���>��T�]       a[��	a	+_��A�*O

prediction_loss���>

reg_loss<�w<


total_loss	��>


accuracy_1�?{��]       a[��	,�	+_��A�*O

prediction_loss�?

reg_loss0�w<


total_lossu�?


accuracy_1���>k�S�]       a[��	��	+_��A�*O

prediction_loss���>

reg_loss&�w<


total_loss��>


accuracy_1�?߯��]       a[��	��	+_��A�*O

prediction_loss{�>

reg_loss�w<


total_loss�ѵ>


accuracy_1��(?��qd]       a[��	��	+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss-?


accuracy_1��>ܹW�]       a[��	p
+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>A{ۙ]       a[��	)&
+_��A�*O

prediction_loss��?

reg_loss��w<


total_lossVx?


accuracy_1���>��w5]       a[��	�>
+_��A�*O

prediction_loss�?

reg_loss�w<


total_losst�?


accuracy_1���>%�']       a[��	�X
+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>�g�&]       a[��	�w
+_��A�*O

prediction_loss   ?

reg_lossܮw<


total_loss��?


accuracy_1   ?�0�]       a[��	�
+_��A�*O

prediction_loss�G�>

reg_lossЮw<


total_loss$�>


accuracy_1)\?2��]       a[��	��
+_��A�*O

prediction_loss���>

reg_lossǮw<


total_lossC��>


accuracy_1��?��]       a[��	t�
+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss$�>


accuracy_1)\?�i�K]       a[��	S�
+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?
�]       a[��	�	+_��A�*O

prediction_loss333?

reg_loss��w<


total_loss�7?


accuracy_1���>����]       a[��	B'+_��A�*O

prediction_loss���>

reg_loss��w<


total_lossB��>


accuracy_1��?�M��]       a[��	�O+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?����]       a[��	�n+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss+?


accuracy_1��>�r��]       a[��	w�+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>�]xa]       a[��	ߤ+_��A�*O

prediction_loss�Q�>

reg_lossu�w<


total_loss`�>


accuracy_1
�#?aG:]       a[��	��+_��A�*O

prediction_loss�Q�>

reg_lossk�w<


total_loss_�>


accuracy_1
�#? at�]       a[��	�+_��A�*O

prediction_loss�z?

reg_lossa�w<


total_loss�Y?


accuracy_1=
�>J-��]       a[��	d+_��A�*O

prediction_loss   ?

reg_lossY�w<


total_loss��?


accuracy_1   ?�9(R]       a[��	G+_��A�*O

prediction_loss)\?

reg_lossM�w<


total_loss�:?


accuracy_1�G�>=���]       a[��	>+_��A�*O

prediction_loss   ?

reg_lossE�w<


total_loss��?


accuracy_1   ?bA�]       a[��	X+_��A�*O

prediction_loss���>

reg_loss8�w<


total_loss��>


accuracy_1�?�}��]       a[��	�n+_��A�*O

prediction_lossR�?

reg_loss.�w<


total_loss�"?


accuracy_1\��>��X�]       a[��	��+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss)?


accuracy_1��>��:�]       a[��	|�+_��A�*O

prediction_loss�?

reg_loss�w<


total_lossp�?


accuracy_1���>��K]       a[��	t
+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss)?


accuracy_1��># \t]       a[��	�7+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?��,]       a[��	&U+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?�_]       a[��	uv+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss)?


accuracy_1��>젺Y]       a[��	��+_��A�*O

prediction_loss)\?

reg_lossޭw<


total_loss�:?


accuracy_1�G�>�i^M]       a[��	+�+_��A�*O

prediction_lossq=
?

reg_lossӭw<


total_loss(?


accuracy_1��>�e�]       a[��	��+_��A�*O

prediction_loss
�#?

reg_lossɭw<


total_loss��'?


accuracy_1�Q�>O�]       a[��	e�+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�I%�]       a[��	+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss;��>


accuracy_1��?���]       a[��	�8+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�B�>


accuracy_1q=
?�z�]       a[��	�X+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�B�>


accuracy_1q=
?Nds�]       a[��	�{+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?��M]       a[��	��+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?3Zw�]       a[��	Q�+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss'?


accuracy_1��>��n�]       a[��	�+_��A�*O

prediction_loss�G�>

reg_lossu�w<


total_loss�>


accuracy_1)\?�w�]       a[��	�+_��A�*O

prediction_lossq=
?

reg_lossl�w<


total_loss'?


accuracy_1��>�9g]       a[��	/+_��A�*O

prediction_loss�?

reg_lossa�w<


total_lossn�?


accuracy_1���>RA�t]       a[��	�O+_��A�*O

prediction_loss)\?

reg_lossV�w<


total_loss�:?


accuracy_1�G�>>&od]       a[��	q+_��A�*O

prediction_lossR�?

reg_lossL�w<


total_loss�"?


accuracy_1\��>�X��]       a[��	ő+_��A�*O

prediction_lossq=
?

reg_lossB�w<


total_loss&?


accuracy_1��>Y��]       a[��	�+_��A�*O

prediction_loss���>

reg_loss8�w<


total_loss��>


accuracy_1�?�3N�]       a[��	A�+_��A�*O

prediction_lossq=
?

reg_loss.�w<


total_loss&?


accuracy_1��>A��~]       a[��	8�+_��A�*O

prediction_loss   ?

reg_loss#�w<


total_loss��?


accuracy_1   ?�D�D]       a[��	i+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?^���]       a[��	Vb+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?�2�(]       a[��	w�+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?n஘]       a[��	��+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>\�A]       a[��	K�+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss�>


accuracy_1)\?=Ԣz]       a[��	X�+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss�>


accuracy_1)\?^��]       a[��	�	+_��A�*O

prediction_loss���>

reg_loss٬w<


total_loss��>


accuracy_1�?�S��]       a[��	o,+_��A�*O

prediction_loss��>

reg_lossϬw<


total_loss�B�>


accuracy_1q=
?�8�]       a[��	�Q+_��A�*O

prediction_loss   ?

reg_lossƬw<


total_loss��?


accuracy_1   ?�z��]       a[��	�s+_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss�"?


accuracy_1\��>��%f]       a[��	>�+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss3��>


accuracy_1��?,���]       a[��	��+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss�>


accuracy_1)\?����]       a[��	M�+_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�L�>


accuracy_1R�?j���]       a[��	&�+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss�>


accuracy_1)\?>0g|]       a[��	�+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss#?


accuracy_1��>B�/i]       a[��	�?+_��A�*O

prediction_lossq=
?

reg_loss{�w<


total_loss#?


accuracy_1��>{^5�]       a[��	�a+_��A�*O

prediction_loss��>

reg_lossq�w<


total_loss�B�>


accuracy_1q=
?ǸL�]       a[��	�+_��A�*O

prediction_loss�?

reg_lossf�w<


total_lossj�?


accuracy_1���>�ں�]       a[��	H�+_��A�*O

prediction_lossR�?

reg_loss\�w<


total_loss�"?


accuracy_1\��>��}�]       a[��	��+_��A�*O

prediction_loss��(?

reg_lossP�w<


total_losst�,?


accuracy_1{�>&QO�]       a[��	a�+_��A�*O

prediction_loss)\?

reg_lossG�w<


total_loss�:?


accuracy_1�G�>U�*]       a[��	� +_��A�*O

prediction_loss�G�>

reg_loss<�w<


total_loss�>


accuracy_1)\?.oM�]       a[��	�(+_��A�*O

prediction_loss���>

reg_loss3�w<


total_loss��>


accuracy_1�?D�]       a[��	�J+_��A�*O

prediction_lossq=
?

reg_loss)�w<


total_loss"?


accuracy_1��>��}]       a[��	��+_��A�*O

prediction_lossR�?

reg_loss�w<


total_loss�"?


accuracy_1\��>IeRG]       a[��	�+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?*�|]       a[��	 �+_��A�*O

prediction_loss)\?

reg_loss
�w<


total_loss�:?


accuracy_1�G�>�&�L]       a[��	��+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�y�]       a[��	A+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss!?


accuracy_1��>�<�]       a[��	-+_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Y?


accuracy_1=
�>5[�]       a[��	M+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>�J�]       a[��	Ou+_��A�*O

prediction_loss���>

reg_loss֫w<


total_loss,��>


accuracy_1��?���]       a[��	�+_��A�*O

prediction_loss�?

reg_loss˫w<


total_lossg�?


accuracy_1���>E;��]       a[��	�+_��A�*O

prediction_lossq=
?

reg_loss«w<


total_loss ?


accuracy_1��>69E]       a[��	��+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�[n]       a[��	�+_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss�"?


accuracy_1\��>Cg�c]       a[��	+_��A�*O

prediction_loss�?

reg_loss��w<


total_lossg�?


accuracy_1���>Ax�G]       a[��	�5+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss|B�>


accuracy_1q=
?�&]       a[��	�S+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?��4]       a[��	^i+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�G6O]       a[��	M�+_��A�*O

prediction_loss=
�>

reg_lossv�w<


total_loss���>


accuracy_1�z?�u�m]       a[��	k�+_��A�*O

prediction_loss��?

reg_lossm�w<


total_lossHx?


accuracy_1���>��ȫ]       a[��	�+_��A�*O

prediction_loss   ?

reg_lossb�w<


total_loss��?


accuracy_1   ?�^�!]       a[��	��+_��A�*O

prediction_loss�z?

reg_lossV�w<


total_loss�Y?


accuracy_1=
�>&�g]       a[��	��+_��A�*O

prediction_loss��>

reg_lossM�w<


total_lossyB�>


accuracy_1q=
?�]       a[��	9
+_��A�*O

prediction_loss�G�>

reg_lossC�w<


total_loss�>


accuracy_1)\?}Ek�]       a[��	�#+_��A�*O

prediction_loss)\?

reg_loss9�w<


total_loss�:?


accuracy_1�G�>��f]       a[��	lC+_��A�*O

prediction_loss   ?

reg_loss/�w<


total_loss��?


accuracy_1   ?H��]       a[��	)]+_��A�*O

prediction_loss�G�>

reg_loss%�w<


total_loss�>


accuracy_1)\?��F$]       a[��	�u+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>w�e�]       a[��	��+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?+�<�]       a[��	��+_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�Y?


accuracy_1=
�>���]       a[��	6�+_��A�*O

prediction_loss�?

reg_loss��w<


total_lossd�?


accuracy_1���>���a]       a[��	 �+_��A�*O

prediction_loss��>

reg_loss�w<


total_losswB�>


accuracy_1q=
?^Ou]       a[��	��+_��A�*O

prediction_loss��>

reg_loss�w<


total_lossvB�>


accuracy_1q=
?1�r>]       a[��	+_��A�*O

prediction_loss�?

reg_lossߪw<


total_lossc�?


accuracy_1���>��ז]       a[��	-+_��A�*O

prediction_loss�Q�>

reg_lossҪw<


total_lossC�>


accuracy_1
�#?��~/]       a[��	�M+_��A�*O

prediction_loss���>

reg_lossʪw<


total_loss��>


accuracy_1�?����]       a[��	�j+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>�؀]       a[��	�+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?��m]       a[��	�+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?}�\v]       a[��	�+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>���]       a[��	��+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss�>


accuracy_1)\?
��]       a[��	��+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>>t*�]       a[��	�+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss!��>


accuracy_1��?d7�]       a[��	�%+_��A�*O

prediction_loss)\?

reg_lossy�w<


total_loss�:?


accuracy_1�G�>�("]       a[��	�@+_��A�*O

prediction_loss\��>

reg_losso�w<


total_loss�L�>


accuracy_1R�?�7d�]       a[��	�a+_��A�*O

prediction_loss�z?

reg_lossd�w<


total_loss�Y?


accuracy_1=
�>y�	�]       a[��	�z+_��A�*O

prediction_loss���>

reg_lossZ�w<


total_loss��>


accuracy_1�?KY�]       a[��	̗+_��A�*O

prediction_loss���>

reg_lossN�w<


total_loss��>


accuracy_1�?s͠�]       a[��	 �+_��A�*O

prediction_loss�z?

reg_lossE�w<


total_loss�Y?


accuracy_1=
�>R���]       a[��	c�+_��A�*O

prediction_loss�?

reg_loss9�w<


total_lossa�?


accuracy_1���>i�]�]       a[��	O�+_��A�*O

prediction_loss   ?

reg_loss0�w<


total_loss��?


accuracy_1   ?S>�']       a[��	+_��A�*O

prediction_loss\��>

reg_loss&�w<


total_loss�L�>


accuracy_1R�?F��r]       a[��	y"+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss?


accuracy_1��>��2]       a[��	A�+_��A�*O

prediction_loss�?

reg_loss	�w<


total_loss`�?


accuracy_1���>$h�]       a[��	��+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>���]       a[��	\�+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss`�?


accuracy_1���>mQ}]       a[��	��+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?c=�]       a[��	� +_��A�*O

prediction_loss�z?

reg_lossީw<


total_loss�Y?


accuracy_1=
�>���]       a[��	+_��A�*O

prediction_loss���>

reg_lossթw<


total_loss��>


accuracy_1�?�!']       a[��	0-+_��A�*O

prediction_loss�z?

reg_lossɩw<


total_loss�Y?


accuracy_1=
�>S]��]       a[��	�N+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?��E�]       a[��	4f+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>�R��]       a[��	��+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss_�?


accuracy_1���>琫K]       a[��	�+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>~*�]       a[��	9�+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>ʄ�]       a[��	��+_��A�*O

prediction_loss��>

reg_loss��w<


total_losskB�>


accuracy_1q=
?R�%]       a[��	[�+_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�L�>


accuracy_1R�?fY[�]       a[��	�+_��A�*O

prediction_loss�?

reg_losst�w<


total_loss^�?


accuracy_1���>{��]       a[��	�+_��A�*O

prediction_loss���>

reg_lossk�w<


total_loss��>


accuracy_1��?O;xC]       a[��	�9+_��A�*O

prediction_loss�?

reg_loss`�w<


total_loss^�?


accuracy_1���>�ڌL]       a[��	�\+_��A�*O

prediction_loss�?

reg_lossX�w<


total_loss]�?


accuracy_1���>�ǔ ]       a[��	��+_��A�*O

prediction_loss�z?

reg_lossL�w<


total_loss�Y?


accuracy_1=
�>6�w]       a[��	"�+_��A�*O

prediction_loss��>

reg_lossD�w<


total_lossiB�>


accuracy_1q=
?A-k]       a[��	��+_��A�*O

prediction_loss{.?

reg_loss:�w<


total_loss �1?


accuracy_1
ף>�m&]       a[��	�s+_��A�*O

prediction_loss��>

reg_loss.�w<


total_losshB�>


accuracy_1q=
?c��]       a[��	ٖ+_��A�*O

prediction_loss��>

reg_loss#�w<


total_losshB�>


accuracy_1q=
?w�&�]       a[��	
�+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?�^�]       a[��	b�+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss\�?


accuracy_1���>$~�]       a[��	��+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?~�mi]       a[��	n+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�L?�]       a[��	1+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?Z��]       a[��	�L+_��A�*O

prediction_loss��?

reg_loss�w<


total_loss>x?


accuracy_1���>}s�]       a[��	�h+_��A�*O

prediction_loss��>

reg_lossڨw<


total_lossfB�>


accuracy_1q=
?_	s�]       a[��	'�+_��A�*O

prediction_loss��?

reg_lossΨw<


total_loss=x?


accuracy_1���>/p�I]       a[��	��+_��A�*O

prediction_lossq=
?

reg_lossŨw<


total_loss?


accuracy_1��>=��]       a[��	8�+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?����]       a[��	V�+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss[�?


accuracy_1���>�3 ]       a[��	��+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>��е]       a[��	�+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?u���]       a[��	p%+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?�ſ]       a[��	JA+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�Y?


accuracy_1=
�>����]       a[��	l[+_��A�*O

prediction_loss�z?

reg_loss|�w<


total_loss�Y?


accuracy_1=
�>	P�X]       a[��	�t+_��A�*O

prediction_loss�z?

reg_losst�w<


total_loss�Y?


accuracy_1=
�>��	]       a[��	i�+_��A�*O

prediction_loss   ?

reg_lossi�w<


total_loss��?


accuracy_1   ?�^f�]       a[��	�+_��A�*O

prediction_loss=
�>

reg_loss_�w<


total_loss���>


accuracy_1�z?��Vl]       a[��	��+_��A�*O

prediction_lossq=
?

reg_lossU�w<


total_loss?


accuracy_1��>	w�U]       a[��	x�+_��A�*O

prediction_loss)\?

reg_lossK�w<


total_loss�:?


accuracy_1�G�>SCԼ]       a[��	��+_��A�*O

prediction_loss���>

reg_lossA�w<


total_loss��>


accuracy_1�?��%]       a[��	�+_��A�*O

prediction_loss��>

reg_loss4�w<


total_lossaB�>


accuracy_1q=
?R?f]       a[��	-'+_��A�*O

prediction_loss���>

reg_loss,�w<


total_loss��>


accuracy_1��?��\t]       a[��	K+_��A�*O

prediction_loss���>

reg_loss!�w<


total_loss��>


accuracy_1�?���]       a[��	zl+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?>'y]       a[��	�+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>�L+]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?�J�]       a[��	��+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?WGp�]       a[��	��+_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss�"?


accuracy_1\��>*��%]       a[��	��+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?�`",]       a[��	Z +_��A�*O

prediction_loss�G�>

reg_lossۧw<


total_loss��>


accuracy_1)\?X.��]       a[��	A( +_��A�*O

prediction_loss��>

reg_lossϧw<


total_loss]B�>


accuracy_1q=
?j�7]       a[��	�A +_��A�*O

prediction_loss��>

reg_lossŧw<


total_loss]B�>


accuracy_1q=
?���]       a[��	�Z +_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1��?�wa]       a[��	Gu +_��A�*O

prediction_loss��?

reg_loss��w<


total_loss9x?


accuracy_1���>k��]       a[��	� +_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�r��]       a[��	� +_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>�"�"]       a[��	H� +_��A�*O

prediction_loss��>

reg_loss��w<


total_loss\B�>


accuracy_1q=
?5ޭ�]       a[��	U� +_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?����]       a[��	'� +_��A�*O

prediction_loss���>

reg_loss�w<


total_loss	��>


accuracy_1��?Es+"]       a[��	�!+_��A�*O

prediction_loss��?

reg_lossu�w<


total_loss8x?


accuracy_1���>%���]       a[��	@0!+_��A�*O

prediction_loss   ?

reg_lossj�w<


total_loss��?


accuracy_1   ?����]       a[��	
J!+_��A�*O

prediction_lossq=
?

reg_lossa�w<


total_loss?


accuracy_1��>I?I]       a[��	ga!+_��A�*O

prediction_loss��>

reg_lossW�w<


total_lossZB�>


accuracy_1q=
?Ď��]       a[��	Ox!+_��A�*O

prediction_loss�Q�>

reg_lossL�w<


total_loss&�>


accuracy_1
�#?���]       a[��	"�!+_��A�*O

prediction_loss�?

reg_loss@�w<


total_lossU�?


accuracy_1���>>�S{]       a[��	@�!+_��A�*O

prediction_loss���>

reg_loss7�w<


total_loss��>


accuracy_1�?��?�]       a[��	�!+_��A�*O

prediction_loss
ף>

reg_loss-�w<


total_lossC��>


accuracy_1{.?�Jo]       a[��	��!+_��A�*O

prediction_loss�G�>

reg_loss#�w<


total_loss��>


accuracy_1)\?.�d�]       a[��	Z�!+_��A�*O

prediction_loss
�#?

reg_loss�w<


total_loss��'?


accuracy_1�Q�>b�a]       a[��	"+_��A�*O

prediction_loss��>

reg_loss�w<


total_lossWB�>


accuracy_1q=
?՜6]       a[��	'"+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss?


accuracy_1��>y"y�]       a[��	B>"+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossWB�>


accuracy_1q=
?Zh��]       a[��	�U"+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_losst��>


accuracy_1�z?�wW�]       a[��	Hk"+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?� ��]       a[��	o"+_��A�*O

prediction_loss)\?

reg_lossצw<


total_loss�:?


accuracy_1�G�>Z�MU]       a[��	l�"+_��A�*O

prediction_loss)\?

reg_lossΦw<


total_loss�:?


accuracy_1�G�>��K]       a[��	��"+_��A�*O

prediction_loss   ?

reg_lossĦw<


total_loss��?


accuracy_1   ?�Nl]       a[��	��"+_��A�*O

prediction_loss�?

reg_loss��w<


total_lossS�?


accuracy_1���>֟��]       a[��	��"+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?)�C]       a[��	3�"+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossTB�>


accuracy_1q=
?����]       a[��	4#+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>xs>�]       a[��	|)#+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossTB�>


accuracy_1q=
?�`O�]       a[��	�@#+_��A�*O

prediction_loss�?

reg_loss��w<


total_lossR�?


accuracy_1���>y�OQ]       a[��	%X#+_��A�*O

prediction_loss�G�>

reg_lossy�w<


total_loss��>


accuracy_1)\?�N]       a[��	/j#+_��A�*O

prediction_loss�G�>

reg_lossp�w<


total_loss��>


accuracy_1)\?��w]       a[��	��#+_��A�*O

prediction_loss��>

reg_lossd�w<


total_lossRB�>


accuracy_1q=
?ߘ�Z]       a[��	��#+_��A�*O

prediction_loss��>

reg_lossX�w<


total_lossRB�>


accuracy_1q=
?VX$E]       a[��	W�#+_��A�*O

prediction_loss�Q�>

reg_lossQ�w<


total_loss�>


accuracy_1
�#?�7�]       a[��	��#+_��A�*O

prediction_lossq=
?

reg_lossG�w<


total_loss
?


accuracy_1��>U�b]       a[��	~�#+_��A�*O

prediction_lossR�?

reg_loss;�w<


total_loss�"?


accuracy_1\��>]{iI]       a[��	r�#+_��A�*O

prediction_loss��>

reg_loss1�w<


total_lossQB�>


accuracy_1q=
?A�R]       a[��	�$+_��A�*O

prediction_loss��>

reg_loss'�w<


total_lossPB�>


accuracy_1q=
?U�gO]       a[��	�+$+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?jَ�]       a[��	�C$+_��A�*O

prediction_loss{.?

reg_loss�w<


total_loss�1?


accuracy_1
ף>�d��]       a[��	6Z$+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?� �]       a[��	��$+_��A�*O

prediction_loss
�#?

reg_loss��w<


total_loss��'?


accuracy_1�Q�>�ʫ]       a[��	�$+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?E�c�]       a[��	(�$+_��A�*O

prediction_loss�G�>

reg_lossޥw<


total_loss��>


accuracy_1)\?�oi[]       a[��	��$+_��A�*O

prediction_loss�z?

reg_lossեw<


total_lossxY?


accuracy_1=
�>�[�]       a[��	$%+_��A�*O

prediction_loss�G�>

reg_lossʥw<


total_loss��>


accuracy_1)\?��`�]       a[��	�2%+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossMB�>


accuracy_1q=
?��-]       a[��	7P%+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�a0]       a[��	�e%+_��A�*O

prediction_loss�?

reg_loss��w<


total_lossO�?


accuracy_1���>���]       a[��	A}%+_��A�*O

prediction_loss�?

reg_loss��w<


total_lossN�?


accuracy_1���>mSia]       a[��	K�%+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?����]       a[��	Ϊ%+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?�I�R]       a[��	b�%+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>2i9�]       a[��	�%+_��A�*O

prediction_loss\��>

reg_lossu�w<


total_loss�L�>


accuracy_1R�?�O��]       a[��	��%+_��A�*O

prediction_loss)\?

reg_lossl�w<


total_loss�:?


accuracy_1�G�>�9�]       a[��	1&+_��A�*O

prediction_loss��>

reg_lossb�w<


total_lossJB�>


accuracy_1q=
?E%��]       a[��	T&+_��A�*O

prediction_loss   ?

reg_lossY�w<


total_loss��?


accuracy_1   ?�䕜]       a[��	�3&+_��A�*O

prediction_loss���>

reg_lossM�w<


total_loss��>


accuracy_1�?4���]       a[��	'M&+_��A�*O

prediction_loss\��>

reg_lossE�w<


total_loss�L�>


accuracy_1R�?st]       a[��	ca&+_��A�*O

prediction_loss=
�>

reg_loss:�w<


total_lossg��>


accuracy_1�z?�fq]       a[��	�y&+_��A�*O

prediction_loss)\?

reg_loss/�w<


total_loss�:?


accuracy_1�G�>~��]       a[��	��&+_��A�*O

prediction_loss)\?

reg_loss$�w<


total_loss�:?


accuracy_1�G�>ݛ�]       a[��	��&+_��A�*O

prediction_loss�z?

reg_loss�w<


total_lossuY?


accuracy_1=
�>@.� ]       a[��	C�&+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss?


accuracy_1��>+@]       a[��	�'+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?@!�1]       a[��	�7'+_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossuY?


accuracy_1=
�>�hX�]       a[��	Q�'+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>�	�X]       a[��	��'+_��A�*O

prediction_loss�?

reg_loss�w<


total_lossL�?


accuracy_1���>s��]       a[��	��'+_��A�*O

prediction_loss�?

reg_lossޤw<


total_lossK�?


accuracy_1���>�*��]       a[��	@�'+_��A�*O

prediction_loss\��>

reg_lossӤw<


total_loss�L�>


accuracy_1R�?�2o]       a[��	��'+_��A�*O

prediction_loss   ?

reg_lossɤw<


total_loss��?


accuracy_1   ?�*=9]       a[��	�(+_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss�>


accuracy_1
�#?�v5�]       a[��	�5(+_��A�*O

prediction_loss�?

reg_loss��w<


total_lossK�?


accuracy_1���>���]       a[��	�U(+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>G��]       a[��	�o(+_��A�*O

prediction_loss�?

reg_loss��w<


total_lossJ�?


accuracy_1���>��7<]       a[��	e�(+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>����]       a[��	Ϥ(+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>�Y[�]       a[��	��(+_��A�*O

prediction_loss�z?

reg_loss��w<


total_losssY?


accuracy_1=
�>J���]       a[��	��(+_��A�*O

prediction_loss\��>

reg_lossw�w<


total_loss�L�>


accuracy_1R�?�V�3]       a[��	<�(+_��A�*O

prediction_loss���>

reg_lossn�w<


total_loss��>


accuracy_1�?��4�]       a[��	�)+_��A�*O

prediction_loss   ?

reg_lossc�w<


total_loss��?


accuracy_1   ?`i�]       a[��	[#)+_��A�*O

prediction_loss�z?

reg_lossY�w<


total_lossrY?


accuracy_1=
�>��#�]       a[��	�A)+_��A�*O

prediction_lossq=
?

reg_lossM�w<


total_loss?


accuracy_1��>
ÚL]       a[��	�])+_��A�*O

prediction_loss�G�>

reg_lossD�w<


total_loss��>


accuracy_1)\?�zr]       a[��	�y)+_��A�*O

prediction_loss��?

reg_loss9�w<


total_loss+x?


accuracy_1���>�/|i]       a[��	�)+_��A�*O

prediction_loss333?

reg_loss.�w<


total_loss�7?


accuracy_1���>pY0]       a[��	��)+_��A�*O

prediction_loss=
�>

reg_loss%�w<


total_loss^��>


accuracy_1�z?��@�]       a[��	6�)+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?�22]       a[��	��)+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?Ǯ8�]       a[��	�)+_��A�*O

prediction_loss��?

reg_loss�w<


total_loss*x?


accuracy_1���>�s��]       a[��	*+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss?


accuracy_1��>ݔϲ]       a[��	o**+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss?B�>


accuracy_1q=
?I�4]       a[��	wI*+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss>B�>


accuracy_1q=
?d3�]       a[��	gd*+_��A�*O

prediction_loss   ?

reg_lossޣw<


total_loss��?


accuracy_1   ?��]       a[��	�|*+_��A�*O

prediction_loss
�#?

reg_lossӣw<


total_loss��'?


accuracy_1�Q�>+�	]       a[��	t�*+_��A�*O

prediction_loss�?

reg_lossɣw<


total_lossG�?


accuracy_1���>�J�M]       a[��	��*+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�ʑ�]       a[��	6�*+_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss�"?


accuracy_1\��>���]       a[��	~�*+_��A�*O

prediction_loss\��>

reg_loss��w<


total_lossyL�>


accuracy_1R�?'��]       a[��	�*+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss<B�>


accuracy_1q=
?j���]       a[��	�++_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�$�]       a[��	{-++_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?;M��]       a[��	F++_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?��ժ]       a[��	Vd++_��A�*O

prediction_loss���>

reg_lossv�w<


total_loss��>


accuracy_1�?����]       a[��	=~++_��A�*O

prediction_loss���>

reg_lossj�w<


total_loss��>


accuracy_1�?�'��]       a[��	��++_��A�*O

prediction_loss��>

reg_lossc�w<


total_loss:B�>


accuracy_1q=
?Y�]       a[��	��++_��A�*O

prediction_loss�G�>

reg_lossW�w<


total_loss��>


accuracy_1)\?;�߅]       a[��	��++_��A�*O

prediction_loss���>

reg_lossN�w<


total_loss��>


accuracy_1��?��]       a[��	�++_��A�*O

prediction_loss��>

reg_lossB�w<


total_loss9B�>


accuracy_1q=
?'[;F]       a[��	*,+_��A�*O

prediction_loss��>

reg_loss9�w<


total_loss9B�>


accuracy_1q=
?��/�]       a[��	H,+_��A�*O

prediction_loss   ?

reg_loss.�w<


total_loss��?


accuracy_1   ?r�n�]       a[��	U0,+_��A�*O

prediction_lossq=
?

reg_loss$�w<


total_loss�?


accuracy_1��>S>�]       a[��	�K,+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?��]       a[��	�i,+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>>J�]       a[��	f�,+_��A�*O

prediction_lossR�?

reg_loss�w<


total_lossޖ"?


accuracy_1\��>���]       a[��	$�,+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�Kt�]       a[��	�,+_��A�*O

prediction_loss�?

reg_loss�w<


total_lossD�?


accuracy_1���>���]       a[��	��,+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1��?�J�]       a[��	�-+_��A�*O

prediction_loss�?

reg_lossڢw<


total_lossC�?


accuracy_1���>jH�]       a[��	:-+_��A�*O

prediction_loss\��>

reg_lossТw<


total_lossrL�>


accuracy_1R�?W�5�]       a[��	��-+_��A�*O

prediction_loss�z?

reg_lossǢw<


total_losslY?


accuracy_1=
�>�Z]       a[��	�-+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?%U]       a[��	��-+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossS��>


accuracy_1�z?��w�]       a[��	��-+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>U�M�]       a[��	�".+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?bw*]       a[��	�I.+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossR��>


accuracy_1�z?�F9�]       a[��	c{.+_��A�*O

prediction_loss
�#?

reg_loss��w<


total_loss��'?


accuracy_1�Q�>�a�s]       a[��	��.+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossQ��>


accuracy_1�z?�[']       a[��	U�.+_��A�*O

prediction_loss�?

reg_losst�w<


total_lossB�?


accuracy_1���>#�T]       a[��	��.+_��A�*O

prediction_loss���>

reg_lossk�w<


total_loss���>


accuracy_1��?���]       a[��	q /+_��A�*O

prediction_lossq=
?

reg_lossa�w<


total_loss�?


accuracy_1��>R�^g]       a[��	q:/+_��A�*O

prediction_loss   ?

reg_lossU�w<


total_loss��?


accuracy_1   ?S�-�]       a[��	W/+_��A�*O

prediction_loss=
�>

reg_lossL�w<


total_lossO��>


accuracy_1�z?`j]       a[��	�~/+_��A�*O

prediction_loss{�>

reg_lossA�w<


total_loss�ѵ>


accuracy_1��(?� ��]       a[��	؟/+_��A�*O

prediction_loss   ?

reg_loss7�w<


total_loss��?


accuracy_1   ?]�C ]       a[��	E�/+_��A�*O

prediction_loss)\?

reg_loss-�w<


total_loss�:?


accuracy_1�G�>MF}7]       a[��	4�/+_��A�*O

prediction_loss)\?

reg_loss#�w<


total_loss�:?


accuracy_1�G�>"]��]       a[��	�0+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?�S�L]       a[��	�'0+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?�>�]       a[��	�B0+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss/B�>


accuracy_1q=
?�q`]       a[��	�]0+_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?|�]       a[��	J�0+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss.B�>


accuracy_1q=
?*{y�]       a[��	�0+_��A�*O

prediction_loss   ?

reg_lossۡw<


total_loss��?


accuracy_1   ?�_}I]       a[��	_�0+_��A�*O

prediction_loss�G�>

reg_lossѡw<


total_loss��>


accuracy_1)\?�Z]       a[��	�1+_��A�*O

prediction_loss=
�>

reg_lossǡw<


total_lossK��>


accuracy_1�z?���L]       a[��	n01+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>_��]       a[��	aP1+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>��u�]       a[��	\r1+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss?�?


accuracy_1���>�Qk]       a[��	��1+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?6]P]       a[��	ٴ1+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss>�?


accuracy_1���>��]       a[��	�1+_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?�ݴ�]       a[��	��1+_��A�*O

prediction_loss�?

reg_loss}�w<


total_loss>�?


accuracy_1���>)lo�]       a[��	X2+_��A�*O

prediction_loss   ?

reg_lossr�w<


total_loss��?


accuracy_1   ?@��]       a[��	B2+_��A�*O

prediction_loss��>

reg_lossh�w<


total_loss*B�>


accuracy_1q=
?K@�]       a[��	0g2+_��A�*O

prediction_loss   ?

reg_loss^�w<


total_loss��?


accuracy_1   ?0�]       a[��	�2+_��A�*O

prediction_loss�?

reg_lossS�w<


total_loss=�?


accuracy_1���>�H7�]       a[��	��2+_��A�*O

prediction_loss=
�>

reg_lossH�w<


total_lossG��>


accuracy_1�z?�g�B]       a[��	y�2+_��A�*O

prediction_loss�z?

reg_loss>�w<


total_lossfY?


accuracy_1=
�>�V�]       a[��	\3+_��A�*O

prediction_loss)\?

reg_loss4�w<


total_loss�:?


accuracy_1�G�>x�0>]       a[��	[@3+_��A�*O

prediction_loss�Q�>

reg_loss(�w<


total_loss��>


accuracy_1
�#?�>t]       a[��	��3+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>L�٧]       a[��	��3+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss(B�>


accuracy_1q=
?G�]       a[��	n�3+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss<�?


accuracy_1���>[�2]       a[��	b�3+_��A�*O

prediction_loss���>

reg_loss��w<


total_lossՉ�>


accuracy_1��?�FQ�]       a[��	� 4+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?w�ɼ]       a[��	$C4+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?;��A]       a[��	h4+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss<�?


accuracy_1���>��Ĕ]       a[��	Q�4+_��A�*O

prediction_loss��>

reg_lossՠw<


total_loss&B�>


accuracy_1q=
?����]       a[��	�4+_��A�*O

prediction_loss�?

reg_loss̠w<


total_loss;�?


accuracy_1���>P�7�]       a[��	�4+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossC��>


accuracy_1�z?���]       a[��	5%5+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>~�
]       a[��	�Q5+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?�[֯]       a[��	nn5+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�<]       a[��	u�5+_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossԖ"?


accuracy_1\��>��~�]       a[��	ȱ5+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?zn�Y]       a[��	��5+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�D�]       a[��	��5+_��A�*O

prediction_loss)\?

reg_lossz�w<


total_loss�:?


accuracy_1�G�>���]       a[��	6+_��A�*O

prediction_loss)\?

reg_lossl�w<


total_loss�:?


accuracy_1�G�>����]       a[��	e86+_��A�*O

prediction_loss)\?

reg_lossd�w<


total_loss�:?


accuracy_1�G�>�|��]       a[��	�Z6+_��A�*O

prediction_loss��>

reg_lossZ�w<


total_loss"B�>


accuracy_1q=
?���]       a[��	�z6+_��A�*O

prediction_loss)\?

reg_lossP�w<


total_loss�:?


accuracy_1�G�>��]       a[��	��6+_��A�*O

prediction_loss=
�>

reg_lossF�w<


total_loss?��>


accuracy_1�z?��_i]       a[��	�6+_��A�*O

prediction_loss
ף>

reg_loss;�w<


total_loss��>


accuracy_1{.?�'~�]       a[��	��6+_��A�*O

prediction_loss���>

reg_loss/�w<


total_loss��>


accuracy_1�?�W��]       a[��	-�6+_��A�*O

prediction_loss�?

reg_loss&�w<


total_loss9�?


accuracy_1���>`�]       a[��	y7+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?	X�D]       a[��	[#7+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?�ǳ�]       a[��	 B7+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss��>


accuracy_1�?ʘHN]       a[��	�Y7+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?pO�]       a[��	�r7+_��A�*O

prediction_loss
�#?

reg_loss�w<


total_loss��'?


accuracy_1�Q�>D� ]       a[��	��7+_��A�*O

prediction_loss��?

reg_loss�w<


total_lossx?


accuracy_1���>�B�]       a[��	z�7+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss��?


accuracy_1   ?�L4�]       a[��	��7+_��A�*O

prediction_loss   ?

reg_loss֟w<


total_loss�?


accuracy_1   ?a��]       a[��	��7+_��A�*O

prediction_loss�G�>

reg_lossɟw<


total_loss��>


accuracy_1)\?���d]       a[��	��7+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossB�>


accuracy_1q=
?�T]       a[��	�	8+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?��]       a[��	�(8+_��A�*O

prediction_loss
�#?

reg_loss��w<


total_loss��'?


accuracy_1�Q�>�R`[]       a[��	�K8+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>Ȋ��]       a[��	�d8+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�"��]       a[��	�z8+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>2e�]       a[��	6�8+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?�^~]       a[��	v�8+_��A�*O

prediction_loss   ?

reg_lossw�w<


total_loss~�?


accuracy_1   ? �i�]       a[��	�8+_��A�*O

prediction_loss)\?

reg_lossm�w<


total_loss�:?


accuracy_1�G�>�L�]       a[��	��8+_��A�*O

prediction_lossR�?

reg_lossc�w<


total_lossЖ"?


accuracy_1\��>�Y08]       a[��	J�8+_��A�*O

prediction_loss�?

reg_lossW�w<


total_loss5�?


accuracy_1���>����]       a[��	�9+_��A�*O

prediction_loss�z?

reg_lossM�w<


total_loss^Y?


accuracy_1=
�>v	�]       a[��	E)9+_��A�*O

prediction_lossR�?

reg_lossC�w<


total_lossϖ"?


accuracy_1\��>�'�]       a[��	�B9+_��A�*O

prediction_loss��>

reg_loss9�w<


total_loss�ی>


accuracy_1�p=?�[]       a[��	�W9+_��A�*O

prediction_loss�G�>

reg_loss/�w<


total_loss��>


accuracy_1)\?��q]       a[��	�u9+_��A�*O

prediction_loss   ?

reg_loss%�w<


total_loss}�?


accuracy_1   ?z�P[]       a[��	��9+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss|�?


accuracy_1   ?���w]       a[��	�9+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>��]       a[��	w�9+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?U��]       a[��	��9+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss4�?


accuracy_1���>���]       a[��	��9+_��A�*O

prediction_loss��?

reg_loss�w<


total_lossx?


accuracy_1���>��s]       a[��	=':+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss|�?


accuracy_1   ?j:~]       a[��	�S:+_��A�*O

prediction_loss��>

reg_lossޞw<


total_lossB�>


accuracy_1q=
?�)>]       a[��	Pn:+_��A�*O

prediction_loss�G�>

reg_lossОw<


total_loss��>


accuracy_1)\?b��^]       a[��	v�:+_��A�*O

prediction_loss)\?

reg_lossɞw<


total_loss�:?


accuracy_1�G�>�92]       a[��	ע:+_��A�*O

prediction_loss{.?

reg_loss��w<


total_loss��1?


accuracy_1
ף>᫻�]       a[��	D�:+_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss͖"?


accuracy_1\��>�]       a[��	+�:+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�>


accuracy_1��?��N�]       a[��	��:+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss2�?


accuracy_1���>���m]       a[��	�;+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1�?�s	�]       a[��	c*;+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossB�>


accuracy_1q=
?�,�[]       a[��	 C;+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>���]       a[��	�\;+_��A�*O

prediction_loss��?

reg_lossv�w<


total_lossx?


accuracy_1���>��x]       a[��	�s;+_��A�*O

prediction_loss���>

reg_lossm�w<


total_loss��>


accuracy_1�?�}��]       a[��	��;+_��A�*O

prediction_loss���>

reg_lossb�w<


total_loss��>


accuracy_1�?�׵]       a[��	;�;+_��A�*O

prediction_loss�?

reg_lossV�w<


total_loss1�?


accuracy_1���>~���]       a[��	��;+_��A�*O

prediction_loss�?

reg_lossL�w<


total_loss1�?


accuracy_1���>^��,]       a[��	1�;+_��A�*O

prediction_loss���>

reg_lossB�w<


total_loss���>


accuracy_1��?�J�<]       a[��	a<+_��A�*O

prediction_lossq=
?

reg_loss6�w<


total_loss�?


accuracy_1��>rjlF]       a[��	"S<+_��A�*O

prediction_loss�G�>

reg_loss-�w<


total_loss��>


accuracy_1)\?��W+]       a[��	nn<+_��A�*O

prediction_loss�?

reg_loss#�w<


total_loss1�?


accuracy_1���>��x]       a[��	ҋ<+_��A�*O

prediction_loss   ?

reg_loss�w<


total_lossx�?


accuracy_1   ?�>"�]       a[��	Է<+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss-��>


accuracy_1�z?���
]       a[��	��<+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss0�?


accuracy_1���>=���]       a[��	3�<+_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossYY?


accuracy_1=
�>�j��]       a[��	d"=+_��A�*O

prediction_loss��>

reg_loss�w<


total_lossB�>


accuracy_1q=
?�utd]       a[��	\T=+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>VX��]       a[��	J�=+_��A�*O

prediction_loss)\?

reg_lossѝw<


total_loss�:?


accuracy_1�G�>��[]       a[��	��=+_��A�*O

prediction_loss��>

reg_lossŝw<


total_lossB�>


accuracy_1q=
?{���]       a[��	k>+_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss��>


accuracy_1
�#?�<,�]       a[��	�.>+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?W���]       a[��	�J>+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss/�?


accuracy_1���>@��]       a[��	�>+_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossv�?


accuracy_1   ?H�(]       a[��	�?+_��A�*O

prediction_loss)\�>

reg_loss��w<


total_loss�>


accuracy_1�Q8?�־�]       a[��	�I?+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss{�>


accuracy_1�?jHU]       a[��	˞?+_��A�*O

prediction_loss�?

reg_loss}�w<


total_loss.�?


accuracy_1���>K���]       a[��	/�?+_��A�*O

prediction_loss)\?

reg_losst�w<


total_loss�:?


accuracy_1�G�>�A�^]       a[��	�/A+_��A�*O

prediction_lossq=
?

reg_lossh�w<


total_loss�?


accuracy_1��>6��g]       a[��	h�A+_��A�*O

prediction_loss=
�>

reg_loss]�w<


total_loss(��>


accuracy_1�z?���?]       a[��	=�A+_��A�*O

prediction_loss�G�>

reg_lossT�w<


total_loss��>


accuracy_1)\?�B�]       a[��	�A+_��A�*O

prediction_loss��>

reg_lossH�w<


total_loss	B�>


accuracy_1q=
?�xL�]       a[��	�4B+_��A�*O

prediction_loss���>

reg_loss?�w<


total_loss���>


accuracy_1��?彏y]       a[��	��B+_��A�*O

prediction_loss)\?

reg_loss5�w<


total_loss�:?


accuracy_1�G�>+���]       a[��	.�B+_��A�*O

prediction_loss)\?

reg_loss*�w<


total_loss�:?


accuracy_1�G�>;`�]       a[��	�(C+_��A�*O

prediction_loss���>

reg_loss�w<


total_lossx�>


accuracy_1�?�.�p]       a[��	�RC+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?g&�]       a[��	��F+_��A�*O

prediction_loss\��>

reg_loss�w<


total_lossDL�>


accuracy_1R�?Oz�]       a[��	�G+_��A�*O

prediction_loss���>

reg_loss�w<


total_lossw�>


accuracy_1�?� ]       a[��	6�G+_��A�*O

prediction_loss   ?

reg_loss��w<


total_losst�?


accuracy_1   ?_ɽ9]       a[��	3QH+_��A�*O

prediction_loss��>

reg_loss�w<


total_lossB�>


accuracy_1q=
?��lR]       a[��	��H+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>Ӓ�]       a[��	��H+_��A�*O

prediction_loss\��>

reg_lossٜw<


total_lossCL�>


accuracy_1R�?<B��]       a[��	bI+_��A�*O

prediction_loss\��>

reg_lossќw<


total_lossCL�>


accuracy_1R�?L> �]       a[��	w-I+_��A�*O

prediction_loss�?

reg_lossĜw<


total_loss+�?


accuracy_1���>��y]       a[��	ZGI+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?�9�]       a[��	1_I+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossB�>


accuracy_1q=
?}}ڀ]       a[��	�I+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss"��>


accuracy_1�z?�Z�]       a[��	-�I+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossB�>


accuracy_1q=
?)L�5]       a[��	�I+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossB�>


accuracy_1q=
?�H��]       a[��	�I+_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossr�?


accuracy_1   ?��i�]       a[��	�J+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss!��>


accuracy_1�z?pt�^]       a[��	� J+_��A�*O

prediction_loss���>

reg_lossu�w<


total_losss�>


accuracy_1�?T9��]       a[��	6=J+_��A�*O

prediction_loss�?

reg_lossl�w<


total_loss*�?


accuracy_1���>���]       a[��	O\J+_��A�*O

prediction_loss��?

reg_lossb�w<


total_lossx?


accuracy_1���>��Y�]       a[��	hwJ+_��A�*O

prediction_loss�?

reg_lossX�w<


total_loss)�?


accuracy_1���>���s]       a[��	��J+_��A�*O

prediction_loss=
�>

reg_lossK�w<


total_loss��>


accuracy_1�z?�e�8]       a[��	u�J+_��A�*O

prediction_loss�z?

reg_lossA�w<


total_lossRY?


accuracy_1=
�>Õ�]       a[��	 �J+_��A�*O

prediction_loss   ?

reg_loss7�w<


total_lossq�?


accuracy_1   ?|@Ș]       a[��	T�J+_��A�*O

prediction_loss=
�>

reg_loss-�w<


total_loss��>


accuracy_1�z?�S�]       a[��	P�J+_��A�*O

prediction_loss���>

reg_loss"�w<


total_lossp�>


accuracy_1�?�[W]       a[��	K+_��A�*O

prediction_loss�z?

reg_loss�w<


total_lossQY?


accuracy_1=
�>�,N]       a[��	�0K+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>o__�]       a[��	�HK+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss(�?


accuracy_1���>�/�]       a[��	,cK+_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossp�?


accuracy_1   ?�6}E]       a[��	�~K+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>�)"�]       a[��	[�K+_��A�*O

prediction_loss�Q�>

reg_loss�w<


total_loss��>


accuracy_1
�#?
�`]       a[��	��K+_��A�*O

prediction_loss�z?

reg_lossכw<


total_lossPY?


accuracy_1=
�>#�$9]       a[��	S�K+_��A�*O

prediction_loss�G�>

reg_lossЛw<


total_loss��>


accuracy_1)\?t,�]       a[��	m�K+_��A�*O

prediction_loss��?

reg_lossśw<


total_loss	x?


accuracy_1���>h�]       a[��	n�K+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss	x?


accuracy_1���>�~�]       a[��	�L+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�A�>


accuracy_1q=
?�<�]       a[��	Z,L+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss'�?


accuracy_1���>����]       a[��	EL+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>`�y]       a[��	]L+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?��9�]       a[��	vL+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>!;s]       a[��	.�L+_��A�*O

prediction_loss��>

reg_lossz�w<


total_loss�A�>


accuracy_1q=
?���:]       a[��	��L+_��A�*O

prediction_lossR�?

reg_lossp�w<


total_loss��"?


accuracy_1\��>J���]       a[��	�L+_��A�*O

prediction_loss�Q�>

reg_lossf�w<


total_loss��>


accuracy_1
�#?'���]       a[��	��L+_��A�*O

prediction_loss
�#?

reg_loss[�w<


total_lossw�'?


accuracy_1�Q�>!�Y]       a[��	|�L+_��A�*O

prediction_loss�G�>

reg_lossQ�w<


total_loss��>


accuracy_1)\?T�U�]       a[��	>M+_��A�*O

prediction_loss   ?

reg_lossE�w<


total_lossm�?


accuracy_1   ?���]       a[��	�M+_��A�*O

prediction_loss���>

reg_loss;�w<


total_loss���>


accuracy_1��?aGVm]       a[��	:M+_��A�*O

prediction_loss���>

reg_loss0�w<


total_lossh�>


accuracy_1�?Ǜ�@]       a[��	�WM+_��A�*O

prediction_loss�?

reg_loss&�w<


total_loss%�?


accuracy_1���>���]       a[��	UlM+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>R>s�]       a[��	��M+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss$�?


accuracy_1���>R�]       a[��	p�M+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss$�?


accuracy_1���>8r�@]       a[��	��M+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>/u��]       a[��	t�M+_��A�*O

prediction_loss���>

reg_loss�w<


total_lossg�>


accuracy_1�?�U��]       a[��	��M+_��A�*O

prediction_loss���>

reg_loss�w<


total_lossf�>


accuracy_1�?�nĂ]       a[��	�N+_��A�*O

prediction_loss�G�>

reg_lossܚw<


total_loss��>


accuracy_1)\?�v�]       a[��	2N+_��A�*O

prediction_loss�?

reg_lossԚw<


total_loss#�?


accuracy_1���>�'i]       a[��	5N+_��A�*O

prediction_loss)\?

reg_lossȚw<


total_loss�:?


accuracy_1�G�>��Y]       a[��	�PN+_��A�*O

prediction_loss���>

reg_loss��w<


total_losse�>


accuracy_1�?�]       a[��	�pN+_��A�*O

prediction_loss���>

reg_loss��w<


total_losse�>


accuracy_1�?���+]       a[��	\�N+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss��>


accuracy_1�z?ݖ|]       a[��	.�N+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss"�?


accuracy_1���>{�]       a[��	~�N+_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossj�?


accuracy_1   ?�v^n]       a[��	��N+_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss��"?


accuracy_1\��>Htt]       a[��	X�N+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?��-%]       a[��	�#O+_��A�*O

prediction_lossR�?

reg_lossw�w<


total_loss��"?


accuracy_1\��>���{]       a[��	`O+_��A�*O

prediction_loss�z?

reg_lossi�w<


total_lossKY?


accuracy_1=
�>�j�r]       a[��	{�O+_��A�*O

prediction_loss���>

reg_loss_�w<


total_lossb�>


accuracy_1�?%!h]       a[��	��O+_��A�*O

prediction_loss�Q�>

reg_lossS�w<


total_loss��>


accuracy_1
�#?�DK�]       a[��	7�O+_��A�*O

prediction_lossR�?

reg_lossJ�w<


total_loss��"?


accuracy_1\��>�P]]       a[��	�O+_��A�*O

prediction_loss)\�>

reg_loss?�w<


total_loss��>


accuracy_1�Q8?�7t]       a[��	
P+_��A�*O

prediction_loss�?

reg_loss7�w<


total_loss!�?


accuracy_1���>�|�:]       a[��	�"P+_��A�*O

prediction_loss)\?

reg_loss,�w<


total_loss�:?


accuracy_1�G�>�<I]       a[��	�7P+_��A�*O

prediction_loss�z?

reg_loss �w<


total_lossJY?


accuracy_1=
�>���+]       a[��	�TP+_��A�*O

prediction_loss   ?

reg_loss�w<


total_lossh�?


accuracy_1   ?�-K�]       a[��	�vP+_��A�*O

prediction_loss��(?

reg_loss�w<


total_loss+�,?


accuracy_1{�>;B]       a[��	��P+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>oY]       a[��	V�P+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss��>


accuracy_1�z?UiI]       a[��	@�P+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss}�>


accuracy_1)\?`�!A]       a[��	�!Q+_��A�*O

prediction_loss   ?

reg_loss�w<


total_lossh�?


accuracy_1   ?ݺ��]       a[��	�R+_��A�*O

prediction_loss   ?

reg_lossٙw<


total_lossg�?


accuracy_1   ?��۾]       a[��	�DR+_��A�*O

prediction_loss=
�>

reg_lossЙw<


total_loss��>


accuracy_1�z?�Cm]       a[��	�S+_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossHY?


accuracy_1=
�>[�k']       a[��	<.S+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>��|�]       a[��	�LS+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss���>


accuracy_1��?�l]       a[��	��S+_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossGY?


accuracy_1=
�>WE�]       a[��	X�S+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss\�>


accuracy_1�?�)%�]       a[��	��S+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossz�>


accuracy_1)\?:c�]       a[��	�S+_��A�*O

prediction_loss�G�>

reg_loss}�w<


total_lossz�>


accuracy_1)\?���]       a[��	A-T+_��A�*O

prediction_lossR�?

reg_losst�w<


total_loss��"?


accuracy_1\��>�%�i]       a[��	KsT+_��A�*O

prediction_loss=
�>

reg_lossk�w<


total_loss��>


accuracy_1�z?���]       a[��	�T+_��A�*O

prediction_loss   ?

reg_lossb�w<


total_lossf�?


accuracy_1   ?q��]       a[��	U+_��A�*O

prediction_lossR�?

reg_lossU�w<


total_loss��"?


accuracy_1\��>G�]       a[��	�U+_��A�*O

prediction_loss�G�>

reg_lossM�w<


total_lossx�>


accuracy_1)\?���]       a[��	�.V+_��A�*O

prediction_loss)\?

reg_lossB�w<


total_loss�:?


accuracy_1�G�>�':]       a[��	�kV+_��A�*O

prediction_loss�z?

reg_loss8�w<


total_lossFY?


accuracy_1=
�>��p�]       a[��	S�V+_��A�*O

prediction_lossR�?

reg_loss.�w<


total_loss��"?


accuracy_1\��>-o.+]       a[��	m�V+_��A�*O

prediction_loss���>

reg_loss$�w<


total_lossX�>


accuracy_1�?��� ]       a[��	N�V+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>a\��]       a[��	��V+_��A�*O

prediction_loss   ?

reg_loss�w<


total_lossd�?


accuracy_1   ?c���]       a[��	,W+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss�?


accuracy_1���>��P�]       a[��	�-W+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>(�^6]       a[��	�QW+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss�:?


accuracy_1�G�>U�m�]       a[��	��W+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>{���]       a[��	��W+_��A�*O

prediction_loss���>

reg_lossژw<


total_lossV�>


accuracy_1�?��q]       a[��	��W+_��A�*O

prediction_loss)\?

reg_lossϘw<


total_loss�:?


accuracy_1�G�>�4��]       a[��	VX+_��A�*O

prediction_loss�G�>

reg_lossw<


total_losst�>


accuracy_1)\?����]       a[��	�'X+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_losst�>


accuracy_1)\?*!%]       a[��	�GX+_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossDY?


accuracy_1=
�>��Z]       a[��	�oX+_��A�*O

prediction_loss���>

reg_loss��w<


total_lossT�>


accuracy_1�?�d�]       a[��	��X+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�A�>


accuracy_1q=
?�6x�]       a[��	��X+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�w?


accuracy_1���>���l]       a[��	��X+_��A�*O

prediction_loss{�>

reg_loss��w<


total_loss?ѵ>


accuracy_1��(?i�]       a[��	O�X+_��A�*O

prediction_loss��>

reg_loss}�w<


total_loss�A�>


accuracy_1q=
?�R �]       a[��	tY+_��A�*O

prediction_lossq=
?

reg_losss�w<


total_loss�?


accuracy_1��>D�F�]       a[��	h#Y+_��A�*O

prediction_loss   ?

reg_lossi�w<


total_lossb�?


accuracy_1   ?a��]       a[��	AY+_��A�*O

prediction_loss   ?

reg_loss`�w<


total_lossb�?


accuracy_1   ?8���]       a[��	_]Y+_��A�*O

prediction_loss��>

reg_lossV�w<


total_loss�A�>


accuracy_1q=
?}��3]       a[��	�{Y+_��A�*O

prediction_loss���>

reg_lossK�w<


total_lossQ�>


accuracy_1�?(�M�]       a[��	��Y+_��A�*O

prediction_loss   ?

reg_lossA�w<


total_lossa�?


accuracy_1   ?�0T�]       a[��	%�Y+_��A�*O

prediction_lossR�?

reg_loss6�w<


total_loss��"?


accuracy_1\��>>��#]       a[��	��Y+_��A�*O

prediction_loss   ?

reg_loss+�w<


total_lossa�?


accuracy_1   ?�@D�]       a[��	c�Y+_��A�*O

prediction_loss�?

reg_loss"�w<


total_loss�?


accuracy_1���>�#]       a[��	�Z+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��> lb�]       a[��	'Z+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_lossn�>


accuracy_1)\?#��]       a[��	(HZ+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss`�?


accuracy_1   ?H̬�]       a[��	nZ+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossn�>


accuracy_1)\?�FG]       a[��	i�Z+_��A�*O

prediction_loss{�>

reg_loss�w<


total_loss:ѵ>


accuracy_1��(?�e/]       a[��	��Z+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss`�?


accuracy_1   ?$MJ�]       a[��	s�Z+_��A�*O

prediction_loss   ?

reg_lossܗw<


total_loss_�?


accuracy_1   ?���]       a[��	3�Z+_��A�*O

prediction_loss)\?

reg_lossїw<


total_loss�:?


accuracy_1�G�>)��]       a[��	[+_��A�*O

prediction_loss��>

reg_lossǗw<


total_loss�A�>


accuracy_1q=
?�m�]       a[��	~7[+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossl�>


accuracy_1)\?��L�]       a[��	)Z[+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss_�?


accuracy_1   ?+d�]       a[��	)w[+_��A�*O

prediction_loss���>

reg_loss��w<


total_lossL�>


accuracy_1�?z�]       a[��	Z�[+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�gj]       a[��	��[+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z??0x�]       a[��	�[+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>���3]       a[��	9\+_��A�*O

prediction_loss���>

reg_loss��w<


total_lossK�>


accuracy_1�?�B6]       a[��	�0\+_��A�*O

prediction_lossq=
?

reg_lossu�w<


total_loss�?


accuracy_1��>T�D]       a[��	7T\+_��A�*O

prediction_loss�?

reg_lossm�w<


total_loss�?


accuracy_1���>Z��#]       a[��	Y�\+_��A�*O

prediction_loss=
�>

reg_lossb�w<


total_loss���>


accuracy_1�z?���x]       a[��	��\+_��A�*O

prediction_loss�z?

reg_lossV�w<


total_loss>Y?


accuracy_1=
�>F]x]       a[��	+�\+_��A�*O

prediction_lossq=
?

reg_lossK�w<


total_loss�?


accuracy_1��>����]       a[��	��\+_��A�*O

prediction_loss���>

reg_lossA�w<


total_lossTV�>


accuracy_1333?�d#]       a[��	�]+_��A�*O

prediction_loss=
�>

reg_loss7�w<


total_loss���>


accuracy_1�z?XF�]       a[��	\9]+_��A�*O

prediction_loss���>

reg_loss/�w<


total_lossH�>


accuracy_1�?$��K]       a[��	XS]+_��A�*O

prediction_loss��?

reg_loss"�w<


total_loss�w?


accuracy_1���>��5]       a[��	r]+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>���r]       a[��	��]+_��A�*O

prediction_loss��?

reg_loss�w<


total_loss�w?


accuracy_1���>��#{]       a[��	�]+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�A�>


accuracy_1q=
?Ժ�]       a[��	�]+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss\�?


accuracy_1   ?r�<�]       a[��	d�]+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�A�>


accuracy_1q=
?bv�W]       a[��	�
^+_��A�*O

prediction_loss\��>

reg_loss�w<


total_lossL�>


accuracy_1R�?����]       a[��	U0^+_��A�*O

prediction_loss�G�>

reg_lossۖw<


total_losse�>


accuracy_1)\?d�]       a[��	KY^+_��A�*O

prediction_loss   ?

reg_lossҖw<


total_loss[�?


accuracy_1   ?�_c�]       a[��	?�^+_��A�*O

prediction_loss��>

reg_lossǖw<


total_loss�A�>


accuracy_1q=
?;�3�]       a[��	�^+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>�8�]]       a[��	|�^+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossd�>


accuracy_1)\?�;��]       a[��	'�^+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss[�?


accuracy_1   ?P�L]       a[��	T_+_��A�*O

prediction_loss��(?

reg_loss��w<


total_loss�,?


accuracy_1{�>���+]       a[��	9_+_��A�*O

prediction_loss���>

reg_loss��w<


total_lossD�>


accuracy_1�?tu}]]       a[��	2V_+_��A�*O

prediction_loss
ף>

reg_loss��w<


total_loss���>


accuracy_1{.?��=�]       a[��	�{_+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>ՙ��]       a[��	��_+_��A�*O

prediction_loss���>

reg_lossu�w<


total_lossC�>


accuracy_1�?.�]]       a[��	_�_+_��A�*O

prediction_loss���>

reg_lossl�w<


total_loss���>


accuracy_1��?	d17]       a[��	�_+_��A�*O

prediction_loss���>

reg_lossb�w<


total_loss���>


accuracy_1��?���]       a[��	�`+_��A�*O

prediction_lossq=
?

reg_lossY�w<


total_loss�?


accuracy_1��>h�]       a[��	�$`+_��A�*O

prediction_loss���>

reg_lossM�w<


total_loss��>


accuracy_1��?H��C]       a[��	�I`+_��A�*O

prediction_loss��>

reg_lossD�w<


total_loss�A�>


accuracy_1q=
?ɧ�J]       a[��	i`+_��A�*O

prediction_loss   ?

reg_loss8�w<


total_lossY�?


accuracy_1   ?�E�d]       a[��	��`+_��A�*O

prediction_loss)\?

reg_loss.�w<


total_loss�:?


accuracy_1�G�>�n�]       a[��	Ϊ`+_��A�*O

prediction_loss���>

reg_loss$�w<


total_loss~��>


accuracy_1��?6�f�]       a[��	C�`+_��A�*O

prediction_lossR�?

reg_loss�w<


total_loss��"?


accuracy_1\��>��zR]       a[��	��`+_��A�*O

prediction_loss{�>

reg_loss�w<


total_loss,ѵ>


accuracy_1��(?�(��]       a[��	�!a+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss�?


accuracy_1���>HU�]       a[��	h@a+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss}��>


accuracy_1��?�F��]       a[��	�aa+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>q�]       a[��	9~a+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss]�>


accuracy_1)\?� f]       a[��	/�a+_��A�*O

prediction_loss�Q8?

reg_lossߕw<


total_lossC0<?


accuracy_1)\�>�_rR]       a[��	r�a+_��A�*O

prediction_loss��>

reg_lossӕw<


total_loss�A�>


accuracy_1q=
?^�.�]       a[��	��a+_��A�*O

prediction_loss��>

reg_lossʕw<


total_loss�A�>


accuracy_1q=
?�\�]       a[��	;ob+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss�:?


accuracy_1�G�>���&]       a[��	��b+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�A�>


accuracy_1q=
?=��j]       a[��	�c+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss�?


accuracy_1���>�պ�]       a[��	�8c+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�w?


accuracy_1���>��]       a[��	 Vc+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�A�>


accuracy_1q=
?��G�]       a[��	Cuc+_��A�*O

prediction_loss���>

reg_loss~�w<


total_loss;�>


accuracy_1�?@V�"]       a[��	}�c+_��A�*O

prediction_loss=
�>

reg_losst�w<


total_loss���>


accuracy_1�z?��l|]       a[��	��c+_��A�*O

prediction_loss��>

reg_lossj�w<


total_loss�A�>


accuracy_1q=
?�]       a[��	9�c+_��A�*O

prediction_loss���>

reg_loss`�w<


total_loss:�>


accuracy_1�?`O�]       a[��	��e+_��A�*O

prediction_loss���>

reg_lossV�w<


total_loss:�>


accuracy_1�?~��]       a[��	\f+_��A�*O

prediction_loss��>

reg_lossK�w<


total_loss�A�>


accuracy_1q=
?����]       a[��	�{f+_��A�*O

prediction_loss�G�>

reg_lossB�w<


total_lossX�>


accuracy_1)\?D��%]       a[��	bNg+_��A�*O

prediction_loss�?

reg_loss7�w<


total_loss�?


accuracy_1���>h���]       a[��	�g+_��A�*O

prediction_loss�G�>

reg_loss-�w<


total_lossW�>


accuracy_1)\?���(]       a[��	Eh+_��A�*O

prediction_loss�G�>

reg_loss"�w<


total_lossW�>


accuracy_1)\?��q�]       a[��	�}h+_��A�*O

prediction_loss   ?

reg_loss�w<


total_lossT�?


accuracy_1   ?c��]       a[��	o�h+_��A�*O

prediction_loss   ?

reg_loss�w<


total_lossT�?


accuracy_1   ?0Uq�]       a[��	1i+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss7�>


accuracy_1�?:�
]       a[��	�Fi+_��A�*O

prediction_loss���>

reg_loss��w<


total_lossu��>


accuracy_1��?�+�]       a[��	�i+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?��3�]       a[��	T�i+_��A�*O

prediction_loss
�#?

reg_loss�w<


total_loss^�'?


accuracy_1�Q�>�&�l]       a[��	�	j+_��A�*O

prediction_lossq=
?

reg_loss۔w<


total_loss�?


accuracy_1��>)SOa]       a[��	4j+_��A�*O

prediction_loss��>

reg_lossДw<


total_loss�A�>


accuracy_1q=
?R#1�]       a[��	�rj+_��A�*O

prediction_loss���>

reg_lossǔw<


total_loss5�>


accuracy_1�?&~ܶ]       a[��	��j+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss4Y?


accuracy_1=
�>���#]       a[��	��j+_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossS�?


accuracy_1   ?*��]       a[��	k+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss4�>


accuracy_1�?�]       a[��	�Nk+_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossR�?


accuracy_1   ?��B]       a[��	�xk+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss4�>


accuracy_1�?5u-]       a[��	~�k+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�A�>


accuracy_1q=
?��]       a[��	��k+_��A�*O

prediction_loss   ?

reg_loss{�w<


total_lossR�?


accuracy_1   ?caJo]       a[��	�l+_��A�*O

prediction_loss=
�>

reg_lossr�w<


total_loss���>


accuracy_1�z?�(��]       a[��	�?l+_��A�*O

prediction_loss���>

reg_lossg�w<


total_loss2�>


accuracy_1�?k�A0]       a[��	�zl+_��A�*O

prediction_loss�?

reg_loss]�w<


total_loss	�?


accuracy_1���>�9k�]       a[��	̸l+_��A�*O

prediction_loss���>

reg_lossR�w<


total_loss2�>


accuracy_1�?E{'�]       a[��	[�l+_��A�*O

prediction_loss)\?

reg_lossG�w<


total_lossz:?


accuracy_1�G�>$���]       a[��	�m+_��A�*O

prediction_lossR�?

reg_loss>�w<


total_loss��"?


accuracy_1\��>J���]       a[��	Bm+_��A�*O

prediction_loss�z?

reg_loss4�w<


total_loss2Y?


accuracy_1=
�>��gG]       a[��	vkm+_��A�*O

prediction_loss��>

reg_loss,�w<


total_loss�A�>


accuracy_1q=
?���]       a[��	ٔm+_��A�*O

prediction_loss   ?

reg_loss"�w<


total_lossQ�?


accuracy_1   ?�a��]       a[��	f�m+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss�?


accuracy_1���>�"9]       a[��	�*n+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?�G��]       a[��	�Rn+_��A�*O

prediction_loss   ?

reg_loss�w<


total_lossP�?


accuracy_1   ?��X]       a[��	~n+_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossP�?


accuracy_1   ?�Xwj]       a[��	��n+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?r.z]       a[��	��n+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>�CM�]       a[��	%o+_��A�*O

prediction_loss)\?

reg_lossՓw<


total_lossx:?


accuracy_1�G�>��{y]       a[��	ao+_��A�*O

prediction_lossq=
?

reg_loss˓w<


total_loss�?


accuracy_1��>�w]       a[��	Y�o+_��A�*O

prediction_loss�?

reg_lossw<


total_loss�?


accuracy_1���>dE5]       a[��	u�o+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>�A�]       a[��	��o+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�A�>


accuracy_1q=
?րZ']       a[��	�5p+_��A�*O

prediction_loss���>

reg_loss��w<


total_lossj��>


accuracy_1��?Yr�]       a[��	�]p+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss/Y?


accuracy_1=
�>�rD]       a[��	p+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossJ�>


accuracy_1)\?�B��]       a[��	��p+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss+�>


accuracy_1�?[�?G]       a[��	�p+_��A�*O

prediction_loss�?

reg_lossw�w<


total_loss�?


accuracy_1���>eM��]       a[��	&q+_��A�*O

prediction_loss   ?

reg_lossm�w<


total_lossN�?


accuracy_1   ?Eb��]       a[��	iTq+_��A�*O

prediction_lossR�?

reg_lossc�w<


total_loss��"?


accuracy_1\��>����]       a[��	��q+_��A�*O

prediction_loss���>

reg_lossX�w<


total_loss*�>


accuracy_1�?تLO]       a[��	��q+_��A�*O

prediction_loss��>

reg_lossM�w<


total_loss�A�>


accuracy_1q=
?�9��]       a[��	��q+_��A�*O

prediction_loss�?

reg_lossD�w<


total_loss�?


accuracy_1���>�J]       a[��	w,r+_��A�*O

prediction_loss���>

reg_loss9�w<


total_lossg��>


accuracy_1��?���!]       a[��	�cr+_��A�*O

prediction_lossR�?

reg_loss/�w<


total_loss��"?


accuracy_1\��>��q�]       a[��	��r+_��A�*O

prediction_loss���>

reg_loss%�w<


total_lossf��>


accuracy_1��?�5Ƶ]       a[��	��r+_��A�*O

prediction_loss��?

reg_loss�w<


total_loss�w?


accuracy_1���>ć�]       a[��	�r+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_lossF�>


accuracy_1)\?��]       a[��	�s+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�A�>


accuracy_1q=
?<��v]       a[��	2;s+_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossL�?


accuracy_1   ?�u�[]       a[��	�ds+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?9��]       a[��	0�s+_��A�*O

prediction_loss\��>

reg_loss�w<


total_loss�K�>


accuracy_1R�?B��]       a[��	��s+_��A�*O

prediction_loss��?

reg_lossܒw<


total_loss�w?


accuracy_1���>
��?]       a[��	h�s+_��A�*O

prediction_loss�G�>

reg_lossҒw<


total_lossE�>


accuracy_1)\?�*�]       a[��	�t+_��A�*O

prediction_loss�?

reg_lossȒw<


total_loss�?


accuracy_1���>��9I]       a[��	�<t+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss,Y?


accuracy_1=
�>��]       a[��	�nt+_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossK�?


accuracy_1   ?
rr]       a[��	E�t+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss$�>


accuracy_1�?h��]       a[��	�t+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�w?


accuracy_1���>��9�]       a[��	��t+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�w?


accuracy_1���>�Z��]       a[��	:u+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�A�>


accuracy_1q=
?uD�]       a[��	�'u+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_lossB�>


accuracy_1)\?zXĎ]       a[��	�Ru+_��A�*O

prediction_loss��>

reg_lossv�w<


total_loss�A�>


accuracy_1q=
?����]       a[��	5zu+_��A�*O

prediction_loss)\?

reg_lossk�w<


total_losss:?


accuracy_1�G�>��Wd]       a[��	��u+_��A�*O

prediction_loss�?

reg_loss`�w<


total_loss�?


accuracy_1���>���]       a[��	��u+_��A�*O

prediction_loss�?

reg_lossU�w<


total_loss�?


accuracy_1���>��^T]       a[��	?�u+_��A�*O

prediction_loss   ?

reg_lossI�w<


total_lossI�?


accuracy_1   ?��I]       a[��	v+_��A�*O

prediction_loss���>

reg_loss?�w<


total_loss!�>


accuracy_1�?���]       a[��	J$v+_��A�*O

prediction_loss)\?

reg_loss5�w<


total_lossr:?


accuracy_1�G�>��"]       a[��	�Qv+_��A�*O

prediction_loss�G�>

reg_loss+�w<


total_loss?�>


accuracy_1)\?Ħ�n]       a[��	�v+_��A�*O

prediction_lossq=
?

reg_loss!�w<


total_loss�?


accuracy_1��>�_�]       a[��	y�v+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss �>


accuracy_1�?�}S�]       a[��	��v+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>��;]       a[��	�w+_��A�*O

prediction_lossR�?

reg_loss�w<


total_loss��"?


accuracy_1\��>�X*']       a[��	�%w+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�w?


accuracy_1���>bb]       a[��	�Rw+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss �?


accuracy_1���>;I�]       a[��	ҋw+_��A�*O

prediction_loss)\?

reg_loss�w<


total_lossq:?


accuracy_1�G�>�а�]       a[��	�w+_��A�*O

prediction_loss�G�>

reg_lossבw<


total_loss=�>


accuracy_1)\?����]       a[��	��w+_��A�*O

prediction_loss��>

reg_loss̑w<


total_loss�A�>


accuracy_1q=
?}u]       a[��	��w+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>$V]       a[��	�x+_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_lossz�>


accuracy_1
�#?
B��]       a[��	�9x+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�6�]       a[��	�x+_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossF�?


accuracy_1   ?�Ψ,]       a[��	~�x+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�>


accuracy_1�?���=]       a[��	J
y+_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�K�>


accuracy_1R�?�$h�]       a[��	�0y+_��A�*O

prediction_loss�?

reg_lossz�w<


total_loss��?


accuracy_1���>W���]       a[��	*Ty+_��A�*O

prediction_loss
�#?

reg_lossp�w<


total_lossP�'?


accuracy_1�Q�>d��}]       a[��	F{y+_��A�*O

prediction_loss���>

reg_lossf�w<


total_loss�>


accuracy_1�?�M��]       a[��	��y+_��A�*O

prediction_lossq=
?

reg_loss\�w<


total_loss�?


accuracy_1��>hy[F]       a[��	��y+_��A�*O

prediction_loss�G�>

reg_lossR�w<


total_loss9�>


accuracy_1)\?	W�"]       a[��	�y+_��A�*O

prediction_lossq=
?

reg_lossH�w<


total_loss�?


accuracy_1��>�x�]       a[��	�z+_��A�*O

prediction_lossq=
?

reg_loss>�w<


total_loss�?


accuracy_1��>���]       a[��	�;z+_��A�*O

prediction_loss=
�>

reg_loss2�w<


total_loss���>


accuracy_1�z?>J��]       a[��	yXz+_��A�*O

prediction_loss���>

reg_loss)�w<


total_lossV��>


accuracy_1��?&�C]       a[��	��z+_��A�*O

prediction_lossq=
?

reg_loss �w<


total_loss�?


accuracy_1��>xF�]       a[��	;�z+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�A�>


accuracy_1q=
?B��(]       a[��	�{+_��A�*O

prediction_loss)\?

reg_loss
�w<


total_lossm:?


accuracy_1�G�>5T��]       a[��	=({+_��A�*O

prediction_loss��>

reg_loss �w<


total_loss�A�>


accuracy_1q=
?�iE]       a[��	�H{+_��A�*O

prediction_loss
ף>

reg_loss��w<


total_loss���>


accuracy_1{.?���]       a[��	�v{+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss5�>


accuracy_1)\?�,�j]       a[��	a�{+_��A�*O

prediction_loss)\?

reg_loss�w<


total_lossm:?


accuracy_1�G�>\�f]       a[��	r�{+_��A�*O

prediction_loss)\?

reg_lossؐw<


total_lossl:?


accuracy_1�G�>,1 =]       a[��	 |+_��A�*O

prediction_loss   ?

reg_lossΐw<


total_lossC�?


accuracy_1   ?\��!]       a[��	�A|+_��A�*O

prediction_lossq=
?

reg_lossw<


total_loss�?


accuracy_1��>��a ]       a[��	�`|+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_loss���>


accuracy_1�z?�c�]       a[��	G�|+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�>


accuracy_1�?��b]       a[��	��|+_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossC�?


accuracy_1   ?���z]       a[��	�}+_��A�*O

prediction_loss   ?

reg_loss��w<


total_lossB�?


accuracy_1   ?���]       a[��	`:}+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�n��]       a[��	t^}+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�>


accuracy_1�?E%w1]       a[��	8�}+_��A�*O

prediction_lossq=
?

reg_lossz�w<


total_loss�?


accuracy_1��>�h�0]       a[��	��}+_��A�*O

prediction_lossq=
?

reg_lossq�w<


total_loss�?


accuracy_1��>	f�\]       a[��	�}+_��A�*O

prediction_loss=
�>

reg_lossf�w<


total_loss���>


accuracy_1�z?`��]       a[��	b~+_��A�*O

prediction_loss��(?

reg_lossY�w<


total_loss�,?


accuracy_1{�>
r�>]       a[��	K~+_��A�*O

prediction_loss�G�>

reg_lossO�w<


total_loss0�>


accuracy_1)\??�J]       a[��	�t~+_��A�*O

prediction_loss�?

reg_lossF�w<


total_loss��?


accuracy_1���>8�6]       a[��	��~+_��A�*O

prediction_loss   ?

reg_loss<�w<


total_lossA�?


accuracy_1   ?��d]       a[��	Ͽ~+_��A�*O

prediction_loss��>

reg_loss1�w<


total_loss�A�>


accuracy_1q=
?1C�]       a[��	-�~+_��A�*O

prediction_loss   ?

reg_loss(�w<


total_lossA�?


accuracy_1   ?/�]       a[��	 +_��A�*O

prediction_loss)\?

reg_loss�w<


total_lossi:?


accuracy_1�G�>�|�
]       a[��	�F+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>X��X]       a[��	Ii+_��A�*O

prediction_lossR�?

reg_loss	�w<


total_loss��"?


accuracy_1\��>X�*�]       a[��	n�+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss!Y?


accuracy_1=
�>~ ��]       a[��	{�+_��A�*O

prediction_loss)\?

reg_loss�w<


total_lossi:?


accuracy_1�G�>��F�]       a[��	/�+_��A�*O

prediction_loss{.?

reg_loss�w<


total_loss��1?


accuracy_1
ף>� s]       a[��	�+_��A�*O

prediction_loss��>

reg_lossޏw<


total_loss�A�>


accuracy_1q=
?�̼]       a[��	��+_��A�*O

prediction_loss���>

reg_lossԏw<


total_loss�>


accuracy_1�?be�b]       a[��	�I�+_��A�*O

prediction_loss��>

reg_lossʏw<


total_loss�A�>


accuracy_1q=
?�B5]       a[��	m�+_��A�*O

prediction_lossR�?

reg_loss��w<


total_loss��"?


accuracy_1\��>���]       a[��	:��+_��A�*O

prediction_loss���>

reg_loss��w<


total_lossK��>


accuracy_1��?dm�(]       a[��	/��+_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossh:?


accuracy_1�G�>�\v]       a[��	�+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�>


accuracy_1�?'� �]       a[��	b,�+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss>�?


accuracy_1   ?s���]       a[��	�s�+_��A�*O

prediction_loss���>

reg_loss��w<


total_lossI��>


accuracy_1��?���]       a[��	:��+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�A�>


accuracy_1q=
?qJ]       a[��	��+_��A�*O

prediction_loss�z?

reg_loss{�w<


total_lossY?


accuracy_1=
�>s��]       a[��	�ہ+_��A�*O

prediction_loss
�#?

reg_lossp�w<


total_lossH�'?


accuracy_1�Q�>��KV]       a[��	���+_��A�*O

prediction_loss   ?

reg_lossg�w<


total_loss>�?


accuracy_1   ?6���]       a[��	�6�+_��A�*O

prediction_loss��>

reg_loss[�w<


total_loss�A�>


accuracy_1q=
?��u�]       a[��	�c�+_��A�*O

prediction_loss)\?

reg_lossP�w<


total_lossf:?


accuracy_1�G�>���]       a[��	���+_��A�*O

prediction_loss=
�>

reg_lossF�w<


total_loss���>


accuracy_1�z?k%S]       a[��	�#�+_��A�*O

prediction_loss�z?

reg_loss<�w<


total_lossY?


accuracy_1=
�>�]��]       a[��	Ec�+_��A�*O

prediction_loss���>

reg_loss1�w<


total_loss	�>


accuracy_1�?*F=�]       a[��	㋃+_��A�*O

prediction_loss���>

reg_loss(�w<


total_loss�>


accuracy_1�?�\]       a[��	���+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>1j�]       a[��	]��+_��A�*O

prediction_loss�z?

reg_loss�w<


total_lossY?


accuracy_1=
�>�� +]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss	�w<


total_loss<�?


accuracy_1   ?�Л�]       a[��	/6�+_��A�*O

prediction_loss
�#?

reg_loss��w<


total_lossF�'?


accuracy_1�Q�>,cFy]       a[��	Ym�+_��A�*O

prediction_loss333?

reg_loss�w<


total_losso7?


accuracy_1���>\E�]       a[��	<��+_��A�*O

prediction_loss
�#?

reg_loss�w<


total_lossF�'?


accuracy_1�Q�>��Ȉ]       a[��	�+_��A�*O

prediction_lossq=
?

reg_lossߎw<


total_loss�?


accuracy_1��>1�3�]       a[��	`�+_��A�*O

prediction_lossq=
?

reg_loss׎w<


total_loss�?


accuracy_1��>ulIc]       a[��	=�+_��A�*O

prediction_lossR�?

reg_loss̎w<


total_loss��"?


accuracy_1\��>��,]       a[��	�/�+_��A�*O

prediction_lossq=
?

reg_lossw<


total_loss�?


accuracy_1��>&��]       a[��	�`�+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss;�?


accuracy_1   ?XD��]       a[��	Ĕ�+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>βG-]       a[��	ϻ�+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�A�>


accuracy_1q=
?7�@]       a[��	�ޅ+_��A�*O

prediction_loss���>

reg_loss��w<


total_lossB��>


accuracy_1��?0a��]       a[��	� �+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�A�>


accuracy_1q=
?�cB�]       a[��	 �+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>Ɣ��]       a[��	XS�+_��A�*O

prediction_loss�G�>

reg_lossz�w<


total_loss"�>


accuracy_1)\?Ԕ<f]       a[��		��+_��A�*O

prediction_loss=
�>

reg_losso�w<


total_loss���>


accuracy_1�z?�Qx
]       a[��	���+_��A�*O

prediction_loss)\?

reg_lossd�w<


total_lossc:?


accuracy_1�G�>���I]       a[��	�҆+_��A�*O

prediction_loss���>

reg_lossY�w<


total_loss@��>


accuracy_1��?��	]       a[��	�+_��A�*O

prediction_loss{�>

reg_lossN�w<


total_loss�е>


accuracy_1��(?��j]       a[��	Y�+_��A�*O

prediction_loss��?

reg_lossD�w<


total_loss�w?


accuracy_1���>5a��]       a[��	]1�+_��A�*O

prediction_loss���>

reg_loss9�w<


total_loss�>


accuracy_1�?b��]       a[��	nh�+_��A�*O

prediction_loss�z?

reg_loss/�w<


total_lossY?


accuracy_1=
�>8M)�]       a[��	�+_��A�*O

prediction_loss��?

reg_loss&�w<


total_loss�w?


accuracy_1���>�P'�]       a[��	.Ƈ+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>�5o�]       a[��	 �+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�A�>


accuracy_1q=
?���D]       a[��	"�+_��A�*O

prediction_loss�z?

reg_loss�w<


total_lossY?


accuracy_1=
�>����]       a[��	�X�+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�w?


accuracy_1���>�KV�]       a[��	�+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss=��>


accuracy_1��?�ȇ�]       a[��	Q��+_��A�*O

prediction_loss)\?

reg_loss�w<


total_lossa:?


accuracy_1�G�>1�g
]       a[��	�#�+_��A�*O

prediction_loss��>

reg_lossݍw<


total_loss�A�>


accuracy_1q=
?'��]       a[��	�G�+_��A�*O

prediction_loss)\?

reg_lossэw<


total_loss`:?


accuracy_1�G�>ޕ�8]       a[��	c|�+_��A�*O

prediction_loss=
�>

reg_lossȍw<


total_loss���>


accuracy_1�z?!:�E]       a[��	���+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�~�>


accuracy_1�?!]       a[��	X�+_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossY?


accuracy_1=
�>�.�w]       a[��	��+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�A�>


accuracy_1q=
?�Ǘ�]       a[��	D0�+_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�K�>


accuracy_1R�?I�Ah]       a[��	���+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>SJ�c]       a[��	��+_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�K�>


accuracy_1R�?���r]       a[��	w/�+_��A�*O

prediction_loss��(?

reg_lossu�w<


total_loss��,?


accuracy_1{�>�}�O]       a[��	rM�+_��A�*O

prediction_loss)\?

reg_lossl�w<


total_loss_:?


accuracy_1�G�>T��]       a[��	�e�+_��A�*O

prediction_loss��>

reg_loss`�w<


total_loss�A�>


accuracy_1q=
?�}:]       a[��	ō�+_��A�*O

prediction_lossq=
?

reg_lossV�w<


total_loss�?


accuracy_1��>�˟�]       a[��	;ȋ+_��A�*O

prediction_loss���>

reg_lossL�w<


total_loss�~�>


accuracy_1�?��]       a[��	%A�+_��A�*O

prediction_loss���>

reg_lossA�w<


total_loss�~�>


accuracy_1�?����]       a[��	�o�+_��A�*O

prediction_loss��?

reg_loss7�w<


total_loss�w?


accuracy_1���>Z�]       a[��	J�+_��A�*O

prediction_loss)\?

reg_loss,�w<


total_loss^:?


accuracy_1�G�>�G�]       a[��	g��+_��A�*O

prediction_loss��>

reg_loss"�w<


total_loss�A�>


accuracy_1q=
?ߐX�]       a[��	s��+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss�A�>


accuracy_1q=
?��r�]       a[��	��+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss]:?


accuracy_1�G�>�#�]       a[��	G��+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?T ��]       a[��	|�+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss�A�>


accuracy_1q=
?�!#]       a[��	%"�+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss�~�>


accuracy_1�?Z]�]       a[��	�O�+_��A�*O

prediction_loss
�#?

reg_loss�w<


total_loss>�'?


accuracy_1�Q�>�	��]       a[��	7l�+_��A�*O

prediction_loss   ?

reg_lossڌw<


total_loss3�?


accuracy_1   ?ٮ=�]       a[��	o��+_��A�*O

prediction_loss�Q�>

reg_lossόw<


total_lossR�>


accuracy_1
�#?�Eԏ]       a[��	w��+_��A�*O

prediction_loss�?

reg_lossČw<


total_loss��?


accuracy_1���>6�B]       a[��	N��+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�~�>


accuracy_1�?��]       a[��		�+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>�})�]       a[��	��+_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_lossQ�>


accuracy_1
�#?�	'�]       a[��	�5�+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>p3ch]       a[��	�L�+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss1��>


accuracy_1��?�B]       a[��	�f�+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss[:?


accuracy_1�G�>�uB�]       a[��	舐+_��A�*O

prediction_loss���>

reg_loss|�w<


total_loss1��>


accuracy_1��?���]       a[��	ʦ�+_��A�*O

prediction_loss�?

reg_lossr�w<


total_loss��?


accuracy_1���>��M�]       a[��	zĐ+_��A�*O

prediction_loss�?

reg_lossg�w<


total_loss��?


accuracy_1���>�Aݴ]       a[��	�ސ+_��A�*O

prediction_loss���>

reg_loss]�w<


total_loss�~�>


accuracy_1�?K�M|]       a[��	F�+_��A�*O

prediction_loss���>

reg_lossR�w<


total_loss0��>


accuracy_1��?���\]       a[��	�:�+_��A�*O

prediction_loss��?

reg_lossI�w<


total_loss�w?


accuracy_1���>�i�.]       a[��	%[�+_��A�*O

prediction_lossR�?

reg_loss@�w<


total_loss��"?


accuracy_1\��>���]       a[��	��+_��A�*O

prediction_loss=
�>

reg_loss5�w<


total_loss���>


accuracy_1�z?ϐ̱]       a[��	A��+_��A�*O

prediction_loss\��>

reg_loss*�w<


total_loss�K�>


accuracy_1R�?�[��]       a[��	���+_��A�*O

prediction_loss)\?

reg_loss�w<


total_lossY:?


accuracy_1�G�>��]       a[��	,ӑ+_��A�*O

prediction_loss\��>

reg_loss�w<


total_loss�K�>


accuracy_1R�?(��]       a[��	��+_��A�*O

prediction_loss�z?

reg_loss�w<


total_lossY?


accuracy_1=
�>;t]       a[��	j�+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss�~�>


accuracy_1�?;a�n]       a[��	9B�+_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�K�>


accuracy_1R�?T�M]       a[��	t]�+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>d�\�]       a[��	���+_��A�*O

prediction_lossR�?

reg_loss�w<


total_loss��"?


accuracy_1\��>�o/�]       a[��	���+_��A�*O

prediction_loss   ?

reg_loss֋w<


total_loss/�?


accuracy_1   ?rmw]       a[��	仒+_��A�*O

prediction_loss   ?

reg_loss̋w<


total_loss/�?


accuracy_1   ?�)p]       a[��	(֒+_��A�*O

prediction_lossq=
?

reg_lossËw<


total_loss�?


accuracy_1��>Z��q]       a[��	9�+_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossX:?


accuracy_1�G�>��+]       a[��	V�+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss�>


accuracy_1)\?��r+]       a[��	�;�+_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_lossI�>


accuracy_1
�#?F�g�]       a[��	4d�+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss|A�>


accuracy_1q=
?��]       a[��	���+_��A�*O

prediction_loss)\?

reg_loss��w<


total_lossW:?


accuracy_1�G�>��+]       a[��	��+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss)��>


accuracy_1��?A*t�]       a[��	��+_��A�*O

prediction_loss\��>

reg_loss{�w<


total_loss�K�>


accuracy_1R�?����]       a[��	�Γ+_��A�*O

prediction_loss
�#?

reg_losss�w<


total_loss8�'?


accuracy_1�Q�>�VM�]       a[��	y�+_��A�*O

prediction_loss�G�>

reg_lossh�w<


total_loss	�>


accuracy_1)\??��P]       a[��	���+_��A�*O

prediction_loss�G�>

reg_loss_�w<


total_loss	�>


accuracy_1)\?6 G�]       a[��	 $�+_��A�*O

prediction_loss�?

reg_lossR�w<


total_loss��?


accuracy_1���>�ZcY]       a[��	�K�+_��A�*O

prediction_loss{�>

reg_lossH�w<


total_loss�е>


accuracy_1��(?���]       a[��	�w�+_��A�*O

prediction_loss�G�>

reg_loss?�w<


total_loss�>


accuracy_1)\?˦X]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss4�w<


total_loss-�?


accuracy_1   ?z*5z]       a[��	���+_��A�*O

prediction_loss�?

reg_loss)�w<


total_loss��?


accuracy_1���>~�r�]       a[��	/ݔ+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss�>


accuracy_1)\?�o�1]       a[��	�+_��A�*O

prediction_loss��>

reg_loss�w<


total_lossxA�>


accuracy_1q=
?����]       a[��	#�+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>xm0]       a[��	YP�+_��A�*O

prediction_loss   ?

reg_loss �w<


total_loss,�?


accuracy_1   ?�2ݒ]       a[��	�p�+_��A�*O

prediction_loss
�#?

reg_loss��w<


total_loss6�'?


accuracy_1�Q�>�ׇ]       a[��	슕+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss,�?


accuracy_1   ?}�l]       a[��	E��+_��A�*O

prediction_loss��>

reg_loss�w<


total_lossvA�>


accuracy_1q=
?f��;]       a[��	{��+_��A�*O

prediction_loss�G�>

reg_loss؊w<


total_loss�>


accuracy_1)\?טg]       a[��	�ו+_��A�*O

prediction_loss=
�>

reg_lossΊw<


total_loss���>


accuracy_1�z?�K(�]       a[��	��+_��A�*O

prediction_lossR�?

reg_lossw<


total_loss}�"?


accuracy_1\��>F�G]       a[��	�
�+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss�>


accuracy_1)\?s�]       a[��	�$�+_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossY?


accuracy_1=
�>C�c�]       a[��	h=�+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�~�>


accuracy_1�?mZJ�]       a[��	6[�+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�w?


accuracy_1���>�Or�]       a[��	>u�+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�~�>


accuracy_1�?e,��]       a[��	���+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�~�>


accuracy_1�?K"s�]       a[��	��+_��A�*O

prediction_lossq=
?

reg_lossz�w<


total_loss�?


accuracy_1��>��Ox]       a[��	Z��+_��A�*O

prediction_loss��?

reg_lossp�w<


total_loss�w?


accuracy_1���>)��]       a[��	�і+_��A�*O

prediction_loss��?

reg_lossf�w<


total_loss�w?


accuracy_1���>�3z�]       a[��	��+_��A�*O

prediction_loss��>

reg_loss[�w<


total_lossrA�>


accuracy_1q=
?�O-W]       a[��	���+_��A�*O

prediction_loss   ?

reg_lossQ�w<


total_loss)�?


accuracy_1   ?1�]       a[��	H�+_��A�*O

prediction_loss   ?

reg_lossG�w<


total_loss)�?


accuracy_1   ?VP�]       a[��	#/�+_��A�*O

prediction_loss�G�>

reg_loss<�w<


total_loss �>


accuracy_1)\?�G �]       a[��	�I�+_��A�*O

prediction_loss)\?

reg_loss3�w<


total_lossR:?


accuracy_1�G�>�Š5]       a[��	�`�+_��A�*O

prediction_loss��>

reg_loss%�w<


total_losspA�>


accuracy_1q=
?"�m�]       a[��	ux�+_��A�*O

prediction_loss)\?

reg_loss�w<


total_lossQ:?


accuracy_1�G�>�,>]       a[��	e��+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?~9�m]       a[��	v��+_��A�*O

prediction_loss���>

reg_loss	�w<


total_loss�~�>


accuracy_1�?���,]       a[��	f��+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss(�?


accuracy_1   ?��^]       a[��	�З+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?�w�]       a[��	�+_��A�*O

prediction_loss\��>

reg_loss�w<


total_loss�K�>


accuracy_1R�?�[�]       a[��	&��+_��A�*O

prediction_loss   ?

reg_loss߉w<


total_loss'�?


accuracy_1   ?�\+�]       a[��	��+_��A�*O

prediction_loss�G�>

reg_lossԉw<


total_loss��>


accuracy_1)\?� ]<]       a[��	�*�+_��A�*O

prediction_loss   ?

reg_lossɉw<


total_loss'�?


accuracy_1   ?<٘(]       a[��	�B�+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�~�>


accuracy_1�?���I]       a[��	�Y�+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�w?


accuracy_1���>�2']       a[��	�p�+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?m�:]       a[��	'��+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�~�>


accuracy_1�?�@!�]       a[��	@��+_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss�K�>


accuracy_1R�?V�˄]       a[��	o��+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>⠣]       a[��	
�+_��A�*O

prediction_loss�G�>

reg_lossy�w<


total_loss��>


accuracy_1)\?u�x]       a[��	O#�+_��A�*O

prediction_loss)\?

reg_lossn�w<


total_lossO:?


accuracy_1�G�>#��]       a[��	:�+_��A�*O

prediction_loss��>

reg_lossc�w<


total_lossjA�>


accuracy_1q=
?��ɚ]       a[��	CT�+_��A�*O

prediction_loss���>

reg_lossY�w<


total_loss��>


accuracy_1��?'8�]       a[��	�y�+_��A�*O

prediction_lossq=
?

reg_lossO�w<


total_loss�?


accuracy_1��>�S�]       a[��	[��+_��A�*O

prediction_loss��>

reg_lossF�w<


total_lossiA�>


accuracy_1q=
?_��8]       a[��	Ů�+_��A�*O

prediction_lossq=
?

reg_loss8�w<


total_loss�?


accuracy_1��>����]       a[��	"ę+_��A�*O

prediction_loss�?

reg_loss0�w<


total_loss��?


accuracy_1���>	�j]       a[��	
֙+_��A�*O

prediction_loss���>

reg_loss&�w<


total_loss�~�>


accuracy_1�?kbv%]       a[��	��+_��A�*O

prediction_lossR�?

reg_loss�w<


total_lossv�"?


accuracy_1\��>"ڥ=]       a[��	:�+_��A�*O

prediction_loss�z?

reg_loss�w<


total_lossY?


accuracy_1=
�>���]       a[��	l!�+_��A�*O

prediction_loss��(?

reg_loss�w<


total_loss��,?


accuracy_1{�>�� ]       a[��	�G�+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossgA�>


accuracy_1q=
?GT�]       a[��	�e�+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>&��]       a[��	���+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>�ɾ]       a[��	M��+_��A�*O

prediction_loss��>

reg_loss߈w<


total_lossfA�>


accuracy_1q=
?���]       a[��	�+_��A�*O

prediction_loss   ?

reg_lossӈw<


total_loss#�?


accuracy_1   ?C-zR]       a[��	ך+_��A�*O

prediction_loss�?

reg_lossʈw<


total_loss��?


accuracy_1���>�N�]       a[��	��+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�w?


accuracy_1���>N覃]       a[��	&�+_��A�*O

prediction_lossR�?

reg_loss��w<


total_lossu�"?


accuracy_1\��>g�7]       a[��	K�+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?5��Q]       a[��	�p�+_��A�*O

prediction_loss�z?

reg_loss��w<


total_lossY?


accuracy_1=
�>��6�]       a[��	{��+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossdA�>


accuracy_1q=
?Wܿ�]       a[��	G �+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?��0�]       a[��	�7�+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>�8]       a[��	k�+_��A�*O

prediction_loss�?

reg_lossv�w<


total_loss��?


accuracy_1���>�B�]       a[��	rݜ+_��A�*O

prediction_loss���>

reg_lossk�w<


total_loss�~�>


accuracy_1�?e���]       a[��	=�+_��A�*O

prediction_loss���>

reg_loss`�w<


total_loss�~�>


accuracy_1�?��pQ]       a[��	�5�+_��A�*O

prediction_loss�z?

reg_lossW�w<


total_lossY?


accuracy_1=
�>�]�]       a[��	�]�+_��A�*O

prediction_lossR�?

reg_lossO�w<


total_losss�"?


accuracy_1\��>���]       a[��	脝+_��A�*O

prediction_loss�?

reg_lossB�w<


total_loss��?


accuracy_1���> �!]       a[��	���+_��A�*O

prediction_lossR�?

reg_loss9�w<


total_losss�"?


accuracy_1\��>�w*]       a[��	�!�+_��A�*O

prediction_lossq=
?

reg_loss.�w<


total_loss�?


accuracy_1��>]�I�]       a[��	�U�+_��A�*O

prediction_loss)\?

reg_loss$�w<


total_lossJ:?


accuracy_1�G�>��<=]       a[��	
��+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss`A�>


accuracy_1q=
?�� �]       a[��	���+_��A�*O

prediction_lossR�?

reg_loss�w<


total_lossr�"?


accuracy_1\��>��b]       a[��	�+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>��]       a[��	�p�+_��A�*O

prediction_loss{�>

reg_loss��w<


total_loss�е>


accuracy_1��(?�S�]       a[��	+��+_��A�*O

prediction_lossR�?

reg_loss�w<


total_lossr�"?


accuracy_1\��>�˔]       a[��	��+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss�~�>


accuracy_1�?��i�]       a[��	�.�+_��A�*O

prediction_loss\��>

reg_loss܇w<


total_loss�K�>


accuracy_1R�?@RE�]       a[��	�T�+_��A�*O

prediction_loss���>

reg_lossчw<


total_loss�~�>


accuracy_1�?�&�?]       a[��	G��+_��A�*O

prediction_loss���>

reg_lossƇw<


total_loss�~�>


accuracy_1�?�Z�]       a[��	z��+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�w?


accuracy_1���>��\�]       a[��	�A�+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�~�>


accuracy_1�?QJ�W]       a[��	���+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss\A�>


accuracy_1q=
? �kP]       a[��	O#�+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossz��>


accuracy_1�z?��]       a[��	�K�+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss
��>


accuracy_1��?�n�r]       a[��	o�+_��A�*O

prediction_loss{�>

reg_loss��w<


total_loss�е>


accuracy_1��(?+츎]       a[��	���+_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�X?


accuracy_1=
�>��"�]       a[��	���+_��A�*O

prediction_loss=
�>

reg_lossu�w<


total_lossy��>


accuracy_1�z?G��]       a[��	��+_��A�*O

prediction_loss���>

reg_lossl�w<


total_loss��>


accuracy_1��?���]       a[��	�G�+_��A�*O

prediction_loss���>

reg_loss`�w<


total_loss�~�>


accuracy_1�?�D6T]       a[��	�o�+_��A�*O

prediction_loss�G�>

reg_lossU�w<


total_loss��>


accuracy_1)\?��P�]       a[��	���+_��A�*O

prediction_loss��>

reg_lossL�w<


total_lossYA�>


accuracy_1q=
?mջ�]       a[��	ܹ�+_��A�*O

prediction_loss{�>

reg_lossA�w<


total_loss�е>


accuracy_1��(?0��l]       a[��	/ߣ+_��A�*O

prediction_loss�z?

reg_loss8�w<


total_loss�X?


accuracy_1=
�>��ӡ]       a[��	�-�+_��A�*O

prediction_loss���>

reg_loss.�w<


total_loss��>


accuracy_1��?P'�]       a[��	�a�+_��A�*O

prediction_loss��>

reg_loss$�w<


total_lossXA�>


accuracy_1q=
?��M�]       a[��	~��+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss�?


accuracy_1   ?1&Ќ]       a[��	)��+_��A�*O

prediction_loss��>

reg_loss�w<


total_lossWA�>


accuracy_1q=
?��$�]       a[��	<ۤ+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss�?


accuracy_1   ?N_]       a[��	`�+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossWA�>


accuracy_1q=
?7� �]       a[��	$�+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>5O��]       a[��	�k�+_��A�*O

prediction_loss��>

reg_loss�w<


total_lossVA�>


accuracy_1q=
?e�B8]       a[��	Û�+_��A�*O

prediction_lossq=
?

reg_lossچw<


total_loss�?


accuracy_1��>�l27]       a[��	�ǥ+_��A�*O

prediction_lossq=
?

reg_lossφw<


total_loss�?


accuracy_1��>��j�]       a[��	��+_��A�*O

prediction_loss���>

reg_lossņw<


total_loss�~�>


accuracy_1�?w~��]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?��Y]       a[��	[�+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossr��>


accuracy_1�z?ԱM]       a[��	۾�+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>����]       a[��	�+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?q`z�]       a[��	��+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�X?


accuracy_1=
�>����]       a[��	1?�+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossSA�>


accuracy_1q=
?G��O]       a[��	=d�+_��A�*O

prediction_lossq=
?

reg_loss|�w<


total_loss�?


accuracy_1��>��Q|]       a[��	���+_��A�*O

prediction_loss)\?

reg_lossr�w<


total_lossC:?


accuracy_1�G�>ϗ�]       a[��	Hާ+_��A�*O

prediction_loss��>

reg_lossg�w<


total_lossRA�>


accuracy_1q=
?���$]       a[��	
�+_��A�*O

prediction_loss��?

reg_loss]�w<


total_loss�w?


accuracy_1���>-x1K]       a[��	�-�+_��A�*O

prediction_loss=
�>

reg_lossR�w<


total_lossp��>


accuracy_1�z?�a�]       a[��	�R�+_��A�*O

prediction_loss�?

reg_lossH�w<


total_loss��?


accuracy_1���>	~1]       a[��	E��+_��A�*O

prediction_loss�z?

reg_loss=�w<


total_loss�X?


accuracy_1=
�>�#z]       a[��	zĨ+_��A�*O

prediction_loss�G�>

reg_loss5�w<


total_loss��>


accuracy_1)\?1���]       a[��	�+_��A�*O

prediction_loss)\?

reg_loss*�w<


total_lossB:?


accuracy_1�G�>1;t5]       a[��	�7�+_��A�*O

prediction_loss��>

reg_loss"�w<


total_lossPA�>


accuracy_1q=
?�u�]       a[��	�Z�+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>	��]       a[��	ׄ�+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss�?


accuracy_1   ?�Z�]       a[��	"��+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>#��]       a[��	��+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�K~�]       a[��	@�+_��A�*O

prediction_loss��>

reg_loss�w<


total_lossNA�>


accuracy_1q=
?��a]       a[��	�T�+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1��?�w�]       a[��	=�+_��A�*O

prediction_loss��>

reg_loss؅w<


total_lossNA�>


accuracy_1q=
?��
C]       a[��	��+_��A�*O

prediction_loss��>

reg_loss̅w<


total_lossMA�>


accuracy_1q=
?EC�J]       a[��	tҪ+_��A�*O

prediction_loss��>

reg_lossÅw<


total_lossMA�>


accuracy_1q=
?T�]       a[��	]�+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?��]]       a[��	�:�+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?ƹ��]       a[��	F[�+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>j&��]       a[��	��+_��A�*O

prediction_lossq=
?

reg_loss��w<


total_loss�?


accuracy_1��>p��]       a[��	r��+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?��\]       a[��	�ի+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss?:?


accuracy_1�G�>���v]       a[��	_�+_��A�*O

prediction_loss��?

reg_loss{�w<


total_loss�w?


accuracy_1���>�78]       a[��	�s�+_��A�*O

prediction_loss�G�>

reg_lossh�w<


total_loss��>


accuracy_1)\?5�]       a[��	���+_��A�*O

prediction_loss��?

reg_loss_�w<


total_loss�w?


accuracy_1���>C\�]       a[��	���+_��A�*O

prediction_lossq=
?

reg_lossR�w<


total_loss�?


accuracy_1��>��Rz]       a[��	NԬ+_��A�*O

prediction_loss��>

reg_lossH�w<


total_lossIA�>


accuracy_1q=
?�9$]       a[��	��+_��A�*O

prediction_loss��>

reg_loss>�w<


total_lossIA�>


accuracy_1q=
?׿!]       a[��	��+_��A�*O

prediction_loss��(?

reg_loss5�w<


total_loss��,?


accuracy_1{�>
��P]       a[��	2>�+_��A�*O

prediction_loss)\?

reg_loss)�w<


total_loss>:?


accuracy_1�G�> ��n]       a[��	�r�+_��A�*O

prediction_loss��?

reg_loss�w<


total_loss�w?


accuracy_1���>��`]       a[��	���+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_lossf��>


accuracy_1�z?]9]       a[��	�ƭ+_��A�*O

prediction_loss   ?

reg_loss
�w<


total_loss�?


accuracy_1   ?��~�]       a[��	x�+_��A�*O

prediction_lossR�?

reg_loss �w<


total_lossf�"?


accuracy_1\��>�w�P]       a[��	�!�+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�U]       a[��	~p�+_��A�*O

prediction_loss{�>

reg_loss�w<


total_loss�е>


accuracy_1��(?��U�]       a[��	Ⱞ+_��A�*O

prediction_loss��?

reg_loss�w<


total_loss�w?


accuracy_1���>���]       a[��	z߮+_��A�*O

prediction_loss�G�>

reg_lossքw<


total_loss��>


accuracy_1)\?�Z�]       a[��	B�+_��A�*O

prediction_loss   ?

reg_loss˄w<


total_loss�?


accuracy_1   ? �]       a[��	���+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�~�>


accuracy_1�?��R]       a[��	�ɯ+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?��~]       a[��	��+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1��?F���]       a[��	�Z�+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�j)]       a[��	,��+_��A�*O

prediction_loss��>

reg_loss��w<


total_lossDA�>


accuracy_1q=
?�;��]       a[��	���+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�w?


accuracy_1���>���g]       a[��	ܰ+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>&�]       a[��	>�+_��A�*O

prediction_loss=
�>

reg_lossz�w<


total_lossa��>


accuracy_1�z?�R��]       a[��	�Y�+_��A�*O

prediction_lossq=
?

reg_lossn�w<


total_loss�?


accuracy_1��>�e��]       a[��	hw�+_��A�*O

prediction_loss=
�>

reg_lossf�w<


total_loss`��>


accuracy_1�z?��)�]       a[��	���+_��A�*O

prediction_loss���>

reg_loss[�w<


total_loss�~�>


accuracy_1�?�&��]       a[��	�ű+_��A�*O

prediction_loss�?

reg_lossQ�w<


total_loss��?


accuracy_1���>����]       a[��	n��+_��A�*O

prediction_loss�?

reg_lossF�w<


total_loss��?


accuracy_1���>~�0�]       a[��	C�+_��A�*O

prediction_loss�?

reg_loss>�w<


total_loss��?


accuracy_1���>G�9�]       a[��	�i�+_��A�*O

prediction_loss���>

reg_loss4�w<


total_loss�~�>


accuracy_1�?~њ2]       a[��	���+_��A�*O

prediction_loss�G�>

reg_loss)�w<


total_loss��>


accuracy_1)\?f�xt]       a[��	9��+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss�?


accuracy_1   ?-j]       a[��	I�+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?��42]       a[��	9*�+_��A�*O

prediction_loss��>

reg_loss
�w<


total_loss?A�>


accuracy_1q=
?S�:]       a[��	�E�+_��A�*O

prediction_lossq=
?

reg_loss �w<


total_loss�?


accuracy_1��>�]�]       a[��	J]�+_��A�*O

prediction_loss\��>

reg_loss��w<


total_loss|K�>


accuracy_1R�?�Jٯ]       a[��	�z�+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss>A�>


accuracy_1q=
?�؂�]       a[��	ݔ�+_��A�*O

prediction_loss��?

reg_loss�w<


total_loss�w?


accuracy_1���>���=]       a[��	#��+_��A�*O

prediction_loss=
�>

reg_loss؃w<


total_loss\��>


accuracy_1�z?�YB�]       a[��	N�+_��A�*O

prediction_loss���>

reg_loss̓w<


total_loss�~�>


accuracy_1�?���]       a[��	��+_��A�*O

prediction_loss�?

reg_lossw<


total_loss��?


accuracy_1���>��R]       a[��	� �+_��A�*O

prediction_loss��?

reg_loss��w<


total_loss�w?


accuracy_1���>��)]       a[��	X9�+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�P�
]       a[��	�S�+_��A�*O

prediction_loss
�#?

reg_loss��w<


total_loss�'?


accuracy_1�Q�>�)]       a[��	jk�+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss7:?


accuracy_1�G�>����]       a[��	���+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?�7N]       a[��	F��+_��A�*O

prediction_loss)\?

reg_loss��w<


total_loss7:?


accuracy_1�G�>ˠ>]       a[��	>��+_��A�*O

prediction_loss��>

reg_lossx�w<


total_loss;A�>


accuracy_1q=
?W�K�]       a[��	d˴+_��A�*O

prediction_loss�G�>

reg_lossk�w<


total_loss��>


accuracy_1)\?�V�~]       a[��	;�+_��A�*O

prediction_lossq=
?

reg_lossb�w<


total_loss?


accuracy_1��>Je� ]       a[��	���+_��A�*O

prediction_loss   ?

reg_lossZ�w<


total_loss�?


accuracy_1   ?N�\]       a[��	H�+_��A�*O

prediction_loss��?

reg_lossQ�w<


total_loss�w?


accuracy_1���>���t]       a[��	'2�+_��A�*O

prediction_loss���>

reg_lossF�w<


total_loss�~�>


accuracy_1�?U���]       a[��	�K�+_��A�*O

prediction_loss���>

reg_loss=�w<


total_loss��>


accuracy_1��?��]       a[��	�c�+_��A�*O

prediction_lossq=
?

reg_loss1�w<


total_loss~?


accuracy_1��>D�8^]       a[��	`x�+_��A�*O

prediction_loss��?

reg_loss'�w<


total_loss�w?


accuracy_1���>��]       a[��	P��+_��A�*O

prediction_loss
�#?

reg_loss�w<


total_loss�'?


accuracy_1�Q�>fjB]       a[��	j��+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss�~�>


accuracy_1�?ˊi	]       a[��	���+_��A�*O

prediction_loss��>

reg_loss
�w<


total_loss7A�>


accuracy_1q=
?�*�H]       a[��	kԵ+_��A�*O

prediction_loss   ?

reg_loss �w<


total_loss�?


accuracy_1   ?��]       a[��	l�+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss�~�>


accuracy_1�?����]       a[��	C�+_��A�*O

prediction_loss��>

reg_loss�w<


total_loss6A�>


accuracy_1q=
?��\�]       a[��	��+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?�:�6]       a[��	#2�+_��A�*O

prediction_loss���>

reg_lossւw<


total_loss�~�>


accuracy_1�?���]       a[��	�J�+_��A�*O

prediction_loss�G�>

reg_loss̂w<


total_loss��>


accuracy_1)\?��l�]       a[��	�a�+_��A�*O

prediction_loss�Q�>

reg_loss��w<


total_loss�>


accuracy_1
�#?$�6�]       a[��	�v�+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss5A�>


accuracy_1q=
?v�.]       a[��	֏�+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>�Hp\]       a[��	��+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>��p%]       a[��	佶+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1��?�v�]       a[��	M׶+_��A�*O

prediction_loss�z?

reg_loss��w<


total_loss�X?


accuracy_1=
�>�\R]       a[��	��+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss��>


accuracy_1��?����]       a[��	5	�+_��A�*O

prediction_loss�?

reg_lossx�w<


total_loss��?


accuracy_1���>�Uu]       a[��	� �+_��A�*O

prediction_loss�?

reg_lossp�w<


total_loss��?


accuracy_1���>�]��]       a[��	8�+_��A�*O

prediction_lossR�?

reg_lossg�w<


total_loss\�"?


accuracy_1\��>o���]       a[��	F\�+_��A�*O

prediction_loss�?

reg_loss\�w<


total_loss��?


accuracy_1���>����]       a[��	�s�+_��A�*O

prediction_loss��?

reg_lossR�w<


total_loss�w?


accuracy_1���>[�]       a[��	�+_��A�*O

prediction_lossq=
?

reg_lossF�w<


total_lossz?


accuracy_1��>�(9�]       a[��	M��+_��A�*O

prediction_loss���>

reg_loss=�w<


total_loss�~�>


accuracy_1�?�Ţ�]       a[��	���+_��A�*O

prediction_loss)\?

reg_loss3�w<


total_loss2:?


accuracy_1�G�>b`��]       a[��	$Է+_��A�*O

prediction_loss{�>

reg_loss(�w<


total_loss�е>


accuracy_1��(?0�l1]       a[��	��+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>��8]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss�?


accuracy_1   ?�ha+]       a[��	i�+_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�X?


accuracy_1=
�>Ȩ�]       a[��	3�+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss݈�>


accuracy_1��?m��]       a[��	J�+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss1:?


accuracy_1�G�>Z]       a[��	�a�+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>�� �]       a[��	�x�+_��A�*O

prediction_loss���>

reg_loss߁w<


total_loss�~�>


accuracy_1�?�β]       a[��	ɐ�+_��A�*O

prediction_loss��>

reg_lossՁw<


total_loss.A�>


accuracy_1q=
?x�`]       a[��	7��+_��A�*O

prediction_loss��?

reg_lossʁw<


total_loss�w?


accuracy_1���>&Z�a]       a[��	D��+_��A�*O

prediction_loss���>

reg_loss��w<


total_lossۈ�>


accuracy_1��?l�8+]       a[��	fڸ+_��A�*O

prediction_loss
�#?

reg_loss��w<


total_loss�'?


accuracy_1�Q�>�[�n]       a[��	��+_��A�*O

prediction_loss=
�>

reg_loss��w<


total_lossJ��>


accuracy_1�z?�d4]       a[��	��+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss�~�>


accuracy_1�?��($]       a[��	4�+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?cj_;]       a[��	�R�+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss+A�>


accuracy_1q=
?< ř]       a[��	!s�+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?,���]       a[��	+_��A�*O

prediction_loss   ?

reg_lossw�w<


total_loss�?


accuracy_1   ?1@�]       a[��	��+_��A�*O

prediction_loss�?

reg_lossk�w<


total_loss��?


accuracy_1���>�z)]       a[��	A�+_��A�*O

prediction_loss�z?

reg_lossW�w<


total_loss�X?


accuracy_1=
�>_�<]       a[��	{2�+_��A�*O

prediction_loss�z?

reg_lossM�w<


total_loss�X?


accuracy_1=
�>�
�R]       a[��	�_�+_��A�*O

prediction_loss�z?

reg_lossA�w<


total_loss�X?


accuracy_1=
�>��>�]       a[��	ڌ�+_��A�*O

prediction_loss��?

reg_loss8�w<


total_loss�w?


accuracy_1���>�J)6]       a[��	a��+_��A�*O

prediction_loss���>

reg_loss-�w<


total_loss�~�>


accuracy_1�?�`GF]       a[��	�˺+_��A�*O

prediction_lossR�?

reg_loss#�w<


total_lossW�"?


accuracy_1\��>o� �]       a[��	Z�+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss-:?


accuracy_1�G�>a�8�]       a[��	�+_��A�*O

prediction_lossq=
?

reg_loss�w<


total_lossu?


accuracy_1��>��"]       a[��	�U�+_��A�*O

prediction_loss�?

reg_loss�w<


total_loss��?


accuracy_1���>~���]       a[��	!u�+_��A�*O

prediction_loss��>

reg_loss��w<


total_loss'A�>


accuracy_1q=
?�0��]       a[��	���+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss�?


accuracy_1   ?}]       a[��	ޫ�+_��A�*O

prediction_loss��?

reg_loss�w<


total_loss�w?


accuracy_1���>���<]       a[��	aû+_��A�*O

prediction_loss���>

reg_lossڀw<


total_loss�~�>


accuracy_1�?��g�]       a[��	�P�+_��A�*O

prediction_loss��?

reg_lossрw<


total_loss�w?


accuracy_1���>�=�]       a[��	 ��+_��A�*O

prediction_loss   ?

reg_lossƀw<


total_loss�?


accuracy_1   ?A���]       a[��	���+_��A�*O

prediction_loss�?

reg_loss��w<


total_loss��?


accuracy_1���>s!ik]       a[��	��+_��A�*O

prediction_loss\��>

reg_loss��w<


total_lossbK�>


accuracy_1R�?�w�]       a[��	J��+_��A�*O

prediction_loss���>

reg_loss��w<


total_loss҈�>


accuracy_1��?�%�#]       a[��	d��+_��A�*O

prediction_loss   ?

reg_loss��w<


total_loss�?


accuracy_1   ?Е��]       a[��	%Ͻ+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?�B>�]       a[��	[	�+_��A�*O

prediction_loss�G�>

reg_loss��w<


total_loss��>


accuracy_1)\?,��]       a[��	O>�+_��A�*O

prediction_loss\��>

reg_loss}�w<


total_loss`K�>


accuracy_1R�?_â�]       a[��	vk�+_��A�*O

prediction_loss��>

reg_lossr�w<


total_loss#A�>


accuracy_1q=
?��GO]       a[��	���+_��A�*O

prediction_lossq=
?

reg_lossg�w<


total_losss?


accuracy_1��>�ݪ�]       a[��	^��+_��A�*O

prediction_loss\��>

reg_loss^�w<


total_loss_K�>


accuracy_1R�?iR�3]       a[��	��+_��A�*O

prediction_loss���>

reg_lossT�w<


total_lossЈ�>


accuracy_1��?��Qx]       a[��	��+_��A�*O

prediction_loss�z?

reg_lossH�w<


total_loss�X?


accuracy_1=
�>���]       a[��	[^�+_��A�*O

prediction_loss   ?

reg_loss=�w<


total_loss�?


accuracy_1   ?��]       a[��	���+_��A�*O

prediction_loss   ?

reg_loss2�w<


total_loss�?


accuracy_1   ?�J|]       a[��	{��+_��A�*O

prediction_loss)\?

reg_loss)�w<


total_loss*:?


accuracy_1�G�>�Е�]       a[��	I��+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss):?


accuracy_1�G�>I�e�]       a[��	gֿ+_��A�*O

prediction_loss�G�>

reg_loss�w<


total_loss��>


accuracy_1)\?�A�]       a[��	��+_��A�*O

prediction_loss�?

reg_loss
�w<


total_loss��?


accuracy_1���>�n]       a[��	tC�+_��A�*O

prediction_loss�G�>

reg_loss �w<


total_loss��>


accuracy_1)\?5�'J]       a[��	@m�+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss͈�>


accuracy_1��?УKA]       a[��	3��+_��A�*O

prediction_loss��>

reg_loss�w<


total_lossA�>


accuracy_1q=
?Ћo)]       a[��	a��+_��A�*O

prediction_loss��?

reg_loss�w<


total_loss�w?


accuracy_1���>"�HD]       a[��	���+_��A�*O

prediction_loss=
�>

reg_loss�w<


total_loss<��>


accuracy_1�z?u"d�]       a[��	���+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?��6�]       a[��	�'�+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss(:?


accuracy_1�G�>X��h]       a[��	\W�+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss�~�>


accuracy_1�?	��]       a[��	t�+_��A�*O

prediction_loss)\?

reg_loss�w<


total_loss(:?


accuracy_1�G�>����]       a[��	���+_��A�*O

prediction_loss���>

reg_loss�w<


total_loss�~�>


accuracy_1�?��I]       a[��	E��+_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�X?


accuracy_1=
�>_nI�]       a[��	���+_��A�*O

prediction_loss�z?

reg_loss�w<


total_loss�X?


accuracy_1=
�>�߻�]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss�w<


total_loss��?


accuracy_1   ?h7�Y]       a[��	/2�+_��A�*O

prediction_loss=
�>

reg_losszw<


total_loss9��>


accuracy_1�z?�.U]       a[��	�S�+_��A�*O

prediction_loss)\?

reg_lossow<


total_loss':?


accuracy_1�G�>RB6�]       a[��	?o�+_��A�*O

prediction_loss�?

reg_lossdw<


total_loss��?


accuracy_1���>j�j�]       a[��	���+_��A�*O

prediction_loss�?

reg_lossZw<


total_loss��?


accuracy_1���>�H}�]       a[��	ݱ�+_��A�*O

prediction_lossq=
?

reg_lossPw<


total_lossn?


accuracy_1��>��u]       a[��	���+_��A�*O

prediction_loss��?

reg_lossFw<


total_loss�w?


accuracy_1���>��u]       a[��	�+_��A�*O

prediction_loss=
�>

reg_loss=w<


total_loss7��>


accuracy_1�z?8B%�]       a[��	��+_��A�*O

prediction_loss��>

reg_loss2w<


total_lossA�>


accuracy_1q=
?yf��]       a[��	C=�+_��A�*O

prediction_loss�G�>

reg_loss'w<


total_loss��>


accuracy_1)\?
kD>]       a[��	�X�+_��A�*O

prediction_loss=
�>

reg_loss w<


total_loss6��>


accuracy_1�z?RY�"]       a[��	�~�+_��A�*O

prediction_loss�?

reg_lossw<


total_loss��?


accuracy_1���>hc�]       a[��	���+_��A�*O

prediction_loss�?

reg_lossw<


total_loss��?


accuracy_1���>�X$]       a[��	��+_��A�*O

prediction_loss�G�>

reg_loss w<


total_loss��>


accuracy_1)\?<��]       a[��	��+_��A�*O

prediction_loss)\?

reg_loss�~w<


total_loss%:?


accuracy_1�G�>Z|�U]       a[��	P9�+_��A�*O

prediction_loss���>

reg_loss�~w<


total_lossĈ�>


accuracy_1��?�4 ]       a[��	�]�+_��A�*O

prediction_loss�G�>

reg_loss�~w<


total_loss��>


accuracy_1)\?\`
�]       a[��	�|�+_��A�*O

prediction_lossq=
?

reg_loss�~w<


total_lossl?


accuracy_1��>W}��]       a[��	O��+_��A�*O

prediction_loss   ?

reg_loss�~w<


total_loss��?


accuracy_1   ?���]       a[��	1��+_��A�*O

prediction_lossq=
?

reg_loss�~w<


total_lossl?


accuracy_1��>���]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss�~w<


total_loss��?


accuracy_1   ?۸�]       a[��	#�+_��A�*O

prediction_lossq=
?

reg_loss�~w<


total_lossl?


accuracy_1��>��f�]       a[��	B�+_��A�*O

prediction_loss��?

reg_loss�~w<


total_loss�w?


accuracy_1���>��ʁ]       a[��	=_�+_��A�*O

prediction_loss{�>

reg_loss�~w<


total_losspе>


accuracy_1��(?}��]       a[��	8��+_��A�*O

prediction_loss���>

reg_loss�~w<


total_loss���>


accuracy_1��?���S]       a[��	*��+_��A�*O

prediction_loss�G�>

reg_loss�~w<


total_loss��>


accuracy_1)\?��AW]       a[��	���+_��A�*O

prediction_loss�?

reg_lossz~w<


total_loss��?


accuracy_1���>�VC�]       a[��	��+_��A�*O

prediction_loss�?

reg_losso~w<


total_loss��?


accuracy_1���>}��|]       a[��	��+_��A�*O

prediction_loss�?

reg_lossf~w<


total_loss��?


accuracy_1���>�H!f]       a[��	�:�+_��A�*O

prediction_loss�?

reg_loss\~w<


total_loss��?


accuracy_1���>��g']       a[��	�i�+_��A�*O

prediction_loss���>

reg_lossR~w<


total_loss�~�>


accuracy_1�?��@�]       a[��	J��+_��A�*O

prediction_loss�Q�>

reg_lossG~w<


total_loss��>


accuracy_1
�#?��-]       a[��	���+_��A�*O

prediction_loss   ?

reg_loss=~w<


total_loss��?


accuracy_1   ?�4��]       a[��	\��+_��A�*O

prediction_loss�G�>

reg_loss3~w<


total_loss��>


accuracy_1)\?���]       a[��	��+_��A�*O

prediction_loss�Q�>

reg_loss+~w<


total_loss��>


accuracy_1
�#?o��f]       a[��	?�+_��A�*O

prediction_loss��>

reg_loss ~w<


total_lossA�>


accuracy_1q=
?I�}�]       a[��	�[�+_��A�*O

prediction_loss�?

reg_loss~w<


total_loss��?


accuracy_1���>�D�]       a[��	t��+_��A�*O

prediction_loss   ?

reg_loss	~w<


total_loss��?


accuracy_1   ?��]       a[��	���+_��A�*O

prediction_loss)\?

reg_loss ~w<


total_loss!:?


accuracy_1�G�>���]       a[��	p	�+_��A�*O

prediction_loss=
�>

reg_loss�}w<


total_loss-��>


accuracy_1�z?��P]       a[��	�\�+_��A�*O

prediction_loss�G�>

reg_loss�}w<


total_loss��>


accuracy_1)\?�mk�]       a[��	�z�+_��A�*O

prediction_loss
�#?

reg_loss�}w<


total_loss�'?


accuracy_1�Q�>Y�`]       a[��	���+_��A�*O

prediction_loss���>

reg_loss�}w<


total_loss~~�>


accuracy_1�?��qx]       a[��	-��+_��A�*O

prediction_loss   ?

reg_loss�}w<


total_loss��?


accuracy_1   ?��u]       a[��	���+_��A�*O

prediction_loss�G�>

reg_loss�}w<


total_loss��>


accuracy_1)\?҄�T]       a[��	��+_��A�*O

prediction_loss�?

reg_loss�}w<


total_loss��?


accuracy_1���>Sr]       a[��	�0�+_��A�*O

prediction_loss�z?

reg_loss�}w<


total_loss�X?


accuracy_1=
�>���b]       a[��	M�+_��A�*O

prediction_loss�G�>

reg_loss�}w<


total_loss��>


accuracy_1)\?�N�]       a[��	Po�+_��A�*O

prediction_loss�?

reg_loss�}w<


total_loss��?


accuracy_1���>!��]       a[��	!��+_��A�*O

prediction_loss�?

reg_loss�}w<


total_loss��?


accuracy_1���>"<\]       a[��	K �+_��A�*O

prediction_loss��?

reg_loss�}w<


total_loss�w?


accuracy_1���>���1]       a[��	�,�+_��A�*O

prediction_loss   ?

reg_lossz}w<


total_loss��?


accuracy_1   ?�d��]       a[��	�\�+_��A�*O

prediction_loss���>

reg_lossp}w<


total_loss���>


accuracy_1��?���b]       a[��	���+_��A�*O

prediction_loss=
�>

reg_lossg}w<


total_loss(��>


accuracy_1�z?xfm�]       a[��	M�+_��A�*O

prediction_loss   ?

reg_lossZ}w<


total_loss��?


accuracy_1   ?	(��]       a[��	CV�+_��A�*O

prediction_loss��>

reg_lossF}w<


total_loss	A�>


accuracy_1q=
?cN�]       a[��	,��+_��A�*O

prediction_loss�z?

reg_loss<}w<


total_loss�X?


accuracy_1=
�>��]       a[��	M��+_��A�*O

prediction_loss�G�>

reg_loss2}w<


total_loss��>


accuracy_1)\?p���]       a[��	j��+_��A�*O

prediction_loss��?

reg_loss(}w<


total_loss�w?


accuracy_1���>��~�]       a[��	���+_��A�*O

prediction_loss=
�>

reg_loss}w<


total_loss&��>


accuracy_1�z?ؠ;	]       a[��	���+_��A�*O

prediction_lossR�?

reg_loss}w<


total_lossF�"?


accuracy_1\��>P��]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss
}w<


total_loss��?


accuracy_1   ?wns]       a[��	cB�+_��A�*O

prediction_loss)\?

reg_loss}w<


total_loss:?


accuracy_1�G�><ZO�]       a[��	�c�+_��A�*O

prediction_lossq=
?

reg_loss�|w<


total_losse?


accuracy_1��>�f�]       a[��	3��+_��A�*O

prediction_loss��>

reg_loss�|w<


total_lossA�>


accuracy_1q=
?K��-]       a[��	]��+_��A�*O

prediction_loss
�#?

reg_loss�|w<


total_loss��'?


accuracy_1�Q�>;�1]       a[��	P��+_��A�*O

prediction_loss�?

reg_loss�|w<


total_loss��?


accuracy_1���>:UO�]       a[��	���+_��A�*O

prediction_loss\��>

reg_loss�|w<


total_lossBK�>


accuracy_1R�?��F]       a[��	Y��+_��A�*O

prediction_lossq=
?

reg_loss�|w<


total_lossd?


accuracy_1��>��gn]       a[��	3�+_��A�*O

prediction_loss�G�>

reg_loss�|w<


total_loss��>


accuracy_1)\?�H]       a[��	�P�+_��A�*O

prediction_loss{�>

reg_loss�|w<


total_loss`е>


accuracy_1��(?��S]       a[��	|�+_��A�*O

prediction_loss�?

reg_loss�|w<


total_loss��?


accuracy_1���>\6@�]       a[��	؛�+_��A�*O

prediction_lossR�?

reg_loss�|w<


total_lossD�"?


accuracy_1\��>f���]       a[��	���+_��A�*O

prediction_loss���>

reg_loss�|w<


total_loss���>


accuracy_1��?!i]       a[��	���+_��A�*O

prediction_loss=
�>

reg_loss�|w<


total_loss!��>


accuracy_1�z?]��O]       a[��	���+_��A�*O

prediction_loss���>

reg_lossz|w<


total_losss~�>


accuracy_1�?}�]       a[��	��+_��A�*O

prediction_loss���>

reg_lossp|w<


total_lossr~�>


accuracy_1�?k�|]       a[��	�+�+_��A�*O

prediction_loss�Q�>

reg_lossd|w<


total_loss��>


accuracy_1
�#?�t�]       a[��	�n�+_��A�*O

prediction_lossq=
?

reg_loss[|w<


total_lossb?


accuracy_1��>8Ηj]       a[��	��+_��A�*O

prediction_loss)\?

reg_lossP|w<


total_loss:?


accuracy_1�G�>/s��]       a[��	��+_��A�*O

prediction_lossq=
?

reg_lossD|w<


total_lossb?


accuracy_1��>�HQ�]       a[��	���+_��A�*O

prediction_loss���>

reg_loss:|w<


total_lossq~�>


accuracy_1�?(ʭ`]       a[��	���+_��A�*O

prediction_loss���>

reg_loss.|w<


total_lossp~�>


accuracy_1�?�]       a[��	6�+_��A�*O

prediction_loss{�>

reg_loss$|w<


total_loss\е>


accuracy_1��(?�O].]       a[��	d"�+_��A�*O

prediction_lossq=
?

reg_loss|w<


total_lossa?


accuracy_1��>@=`�]       a[��	�:�+_��A�*O

prediction_loss�Q�>

reg_loss|w<


total_loss��>


accuracy_1
�#?;ꍴ]       a[��	�c�+_��A�*O

prediction_loss�?

reg_loss|w<


total_loss��?


accuracy_1���>��7�]       a[��	��+_��A�*O

prediction_loss�G�>

reg_loss�{w<


total_loss��>


accuracy_1)\?��9]       a[��	J��+_��A�*O

prediction_loss���>

reg_loss�{w<


total_losso~�>


accuracy_1�?���]       a[��	���+_��A�*O

prediction_loss���>

reg_loss�{w<


total_lossn~�>


accuracy_1�?��]       a[��	���+_��A�*O

prediction_loss��?

reg_loss�{w<


total_loss�w?


accuracy_1���>K=�"]       a[��	
�+_��A�*O

prediction_loss���>

reg_loss�{w<


total_lossn~�>


accuracy_1�?��.B]       a[��	�+�+_��A�*O

prediction_lossq=
?

reg_loss�{w<


total_loss`?


accuracy_1��>دX]       a[��	�D�+_��A�*O

prediction_loss�z?

reg_loss�{w<


total_loss�X?


accuracy_1=
�>���]       a[��	_�+_��A�*O

prediction_loss   ?

reg_loss�{w<


total_loss��?


accuracy_1   ?�.�!]       a[��	N��+_��A�*O

prediction_loss   ?

reg_loss�{w<


total_loss��?


accuracy_1   ?M��]       a[��	��+_��A�*O

prediction_loss�z?

reg_loss�{w<


total_loss�X?


accuracy_1=
�>�bD]       a[��	���+_��A�*O

prediction_lossq=
?

reg_loss�{w<


total_loss_?


accuracy_1��>��]       a[��	���+_��A�*O

prediction_loss�G�>

reg_loss�{w<


total_loss��>


accuracy_1)\?�-+W]       a[��	��+_��A�*O

prediction_loss�G�>

reg_loss�{w<


total_loss��>


accuracy_1)\?��<�]       a[��	s-�+_��A�*O

prediction_loss��?

reg_lossv{w<


total_loss�w?


accuracy_1���>�U]]       a[��	gH�+_��A�*O

prediction_loss�?

reg_lossk{w<


total_loss��?


accuracy_1���>x�^']       a[��	�d�+_��A�*O

prediction_loss   ?

reg_lossa{w<


total_loss��?


accuracy_1   ?P'ш]       a[��	��+_��A�*O

prediction_loss   ?

reg_lossX{w<


total_loss��?


accuracy_1   ?���Y]       a[��	`��+_��A�*O

prediction_loss\��>

reg_lossM{w<


total_loss6K�>


accuracy_1R�?��h]       a[��	��+_��A�*O

prediction_loss��>

reg_lossB{w<


total_loss�@�>


accuracy_1q=
?E`��]       a[��	��+_��A�*O

prediction_loss���>

reg_loss8{w<


total_lossi~�>


accuracy_1�?dU]       a[��	j3�+_��A�*O

prediction_loss��>

reg_loss.{w<


total_loss�@�>


accuracy_1q=
?`'�G]       a[��	IM�+_��A�*O

prediction_loss=
�>

reg_loss${w<


total_loss��>


accuracy_1�z?�~H]       a[��	Eb�+_��A�*O

prediction_loss)\?

reg_loss{w<


total_loss:?


accuracy_1�G�>��]       a[��	w��+_��A�*O

prediction_loss��(?

reg_loss{w<


total_loss��,?


accuracy_1{�>ItQ]       a[��	j��+_��A�*O

prediction_loss�G�>

reg_loss{w<


total_loss��>


accuracy_1)\?)(?F]       a[��	4��+_��A�*O

prediction_loss��>

reg_loss�zw<


total_loss�@�>


accuracy_1q=
?�^v]       a[��	 ��+_��A�*O

prediction_loss�G�>

reg_loss�zw<


total_loss��>


accuracy_1)\?�Uf]       a[��	v�+_��A�*O

prediction_lossR�?

reg_loss�zw<


total_loss>�"?


accuracy_1\��>:�\]       a[��	1�+_��A�*O

prediction_loss�?

reg_loss�zw<


total_loss��?


accuracy_1���>Ƀ�&]       a[��	�W�+_��A�*O

prediction_loss��>

reg_loss�zw<


total_loss�@�>


accuracy_1q=
?[W�c]       a[��	Cs�+_��A�*O

prediction_loss   ?

reg_loss�zw<


total_loss��?


accuracy_1   ?A�|�]       a[��	/��+_��A�*O

prediction_loss�Q�>

reg_loss�zw<


total_loss��>


accuracy_1
�#?6���]       a[��	���+_��A�*O

prediction_loss��>

reg_loss�zw<


total_loss�@�>


accuracy_1q=
?a�]       a[��	���+_��A�*O

prediction_loss   ?

reg_loss�zw<


total_loss��?


accuracy_1   ?auy]       a[��	J�+_��A�*O

prediction_loss�?

reg_loss�zw<


total_loss��?


accuracy_1���>8��]       a[��	%!�+_��A�*O

prediction_loss   ?

reg_loss�zw<


total_loss��?


accuracy_1   ?3��9]       a[��	2;�+_��A�*O

prediction_loss   ?

reg_loss�zw<


total_loss��?


accuracy_1   ?\t�F]       a[��	�X�+_��A�*O

prediction_loss{�>

reg_losszw<


total_lossOе>


accuracy_1��(?*U\�]       a[��	z�+_��A�*O

prediction_loss)\?

reg_losstzw<


total_loss:?


accuracy_1�G�>{g7}]       a[��	g��+_��A�*O

prediction_loss���>

reg_losslzw<


total_lossb~�>


accuracy_1�?�/;x]       a[��	��+_��A�*O

prediction_loss=
�>

reg_loss_zw<


total_loss��>


accuracy_1�z?���m]       a[��	���+_��A�*O

prediction_loss��?

reg_lossSzw<


total_loss�w?


accuracy_1���>l~p]       a[��	"��+_��A�*O

prediction_loss�z?

reg_lossIzw<


total_loss�X?


accuracy_1=
�>����]       a[��	z�+_��A�*O

prediction_loss�G�>

reg_loss@zw<


total_loss��>


accuracy_1)\?fҤ~]       a[��	6�+_��A�*O

prediction_loss�G�>

reg_loss6zw<


total_loss��>


accuracy_1)\?~�}�]       a[��	�W�+_��A�*O

prediction_lossq=
?

reg_loss,zw<


total_lossZ?


accuracy_1��>�>��]       a[��	�t�+_��A�*O

prediction_loss�z?

reg_losszw<


total_loss�X?


accuracy_1=
�>Q�x]       a[��	���+_��A�*O

prediction_loss   ?

reg_losszw<


total_loss��?


accuracy_1   ?�j�]       a[��	��+_��A�*O

prediction_loss�?

reg_losszw<


total_loss��?


accuracy_1���>�B�]       a[��	W��+_��A�*O

prediction_loss�G�>

reg_losszw<


total_loss~�>


accuracy_1)\? �^�]       a[��	���+_��A�*O

prediction_loss���>

reg_loss�yw<


total_loss_~�>


accuracy_1�?�5{�]       a[��	W�+_��A�*O

prediction_loss\��>

reg_loss�yw<


total_loss+K�>


accuracy_1R�?E��X]       a[��	��+_��A�*O

prediction_loss��>

reg_loss�yw<


total_loss�@�>


accuracy_1q=
?��ch]       a[��	^0�+_��A�*O

prediction_loss=
�>

reg_loss�yw<


total_loss��>


accuracy_1�z?�4�]       a[��	�B�+_��A�*O

prediction_lossq=
?

reg_loss�yw<


total_lossX?


accuracy_1��>W��D]       a[��	8g�+_��A�*O

prediction_loss�?

reg_loss�yw<


total_loss��?


accuracy_1���>�)l=]       a[��	w��+_��A�*O

prediction_lossq=
?

reg_loss�yw<


total_lossX?


accuracy_1��>v?��]       a[��	���+_��A�*O

prediction_loss   ?

reg_loss�yw<


total_loss��?


accuracy_1   ?�M��]       a[��	���+_��A�*O

prediction_loss���>

reg_loss�yw<


total_loss���>


accuracy_1��?.+]       a[��	9��+_��A�*O

prediction_loss   ?

reg_loss�yw<


total_loss��?


accuracy_1   ?��=�]       a[��	]�+_��A�*O

prediction_loss   ?

reg_loss�yw<


total_loss��?


accuracy_1   ?9ƨ�]       a[��	xC�+_��A�*O

prediction_loss�G�>

reg_loss�yw<


total_lossz�>


accuracy_1)\?���]       a[��	�d�+_��A�*O

prediction_loss{�>

reg_loss|yw<


total_lossGе>


accuracy_1��(?���]       a[��	U��+_��A�*O

prediction_loss   ?

reg_lossryw<


total_loss��?


accuracy_1   ?6���]       a[��	��+_��A�*O

prediction_loss��>

reg_losshyw<


total_loss�@�>


accuracy_1q=
?����]       a[��	T�+_��A�*O

prediction_lossq=
?

reg_loss]yw<


total_lossV?


accuracy_1��>��2]       a[��	2�+_��A�*O

prediction_loss)\?

reg_lossRyw<


total_loss:?


accuracy_1�G�>�>5�]       a[��	DR�+_��A�*O

prediction_loss   ?

reg_lossHyw<


total_loss��?


accuracy_1   ?3uL�]       a[��	I��+_��A�*O

prediction_loss��>

reg_loss2yw<


total_loss�@�>


accuracy_1q=
?�s2�]       a[��	~�+_��A�*O

prediction_loss���>

reg_loss)yw<


total_lossX~�>


accuracy_1�?MS�]       a[��	�9�+_��A�*O

prediction_loss���>

reg_lossyw<


total_lossX~�>


accuracy_1�?�(��]       a[��	�e�+_��A�*O

prediction_loss   ?

reg_lossyw<


total_loss��?


accuracy_1   ?��v�]       a[��	a��+_��A�*O

prediction_loss   ?

reg_lossyw<


total_loss��?


accuracy_1   ?f2�]       a[��	1��+_��A�*O

prediction_loss\��>

reg_loss�xw<


total_loss$K�>


accuracy_1R�?3;��]       a[��	��+_��A�*O

prediction_loss\��>

reg_loss�xw<


total_loss$K�>


accuracy_1R�?�g�c]       a[��	p@�+_��A�*O

prediction_loss�G�>

reg_loss�xw<


total_lossu�>


accuracy_1)\?��f]       a[��	Lk�+_��A�*O

prediction_loss���>

reg_loss�xw<


total_loss���>


accuracy_1��?X�7]       a[��	)��+_��A�*O

prediction_loss�G�>

reg_loss�xw<


total_lossu�>


accuracy_1)\?�U�0]       a[��	f��+_��A�*O

prediction_loss���>

reg_loss�xw<


total_lossU~�>


accuracy_1�?�8��]       a[��	�!�+_��A�*O

prediction_loss   ?

reg_loss�xw<


total_loss��?


accuracy_1   ?#�x]       a[��	i�+_��A�*O

prediction_loss�z?

reg_loss�xw<


total_loss�X?


accuracy_1=
�>��M�]       a[��	|��+_��A�*O

prediction_loss)\?

reg_loss�xw<


total_loss:?


accuracy_1�G�>�W�o]       a[��	�2�+_��A�*O

prediction_loss�z?

reg_loss�xw<


total_loss�X?


accuracy_1=
�>ʞL]       a[��	l]�+_��A�*O

prediction_loss   ?

reg_loss�xw<


total_loss��?


accuracy_1   ?|��]       a[��	��+_��A�*O

prediction_loss�?

reg_loss�xw<


total_loss��?


accuracy_1���>��Ο]       a[��	ß�+_��A�*O

prediction_loss)\?

reg_loss�xw<


total_loss:?


accuracy_1�G�>��1�]       a[��	��+_��A�*O

prediction_loss��?

reg_lossyxw<


total_loss|w?


accuracy_1���>����]       a[��	*�+_��A�*O

prediction_loss���>

reg_lossoxw<


total_loss���>


accuracy_1��?�T(]       a[��	EI�+_��A�*O

prediction_loss)\?

reg_lossdxw<


total_loss:?


accuracy_1�G�>��j]       a[��	�n�+_��A�*O

prediction_loss�G�>

reg_loss[xw<


total_lossq�>


accuracy_1)\?�!]       a[��	��+_��A�*O

prediction_loss)\?

reg_lossQxw<


total_loss
:?


accuracy_1�G�>i���]       a[��	��+_��A�*O

prediction_loss�?

reg_lossFxw<


total_loss��?


accuracy_1���>��8�]       a[��	��+_��A�*O

prediction_loss=
�>

reg_loss<xw<


total_loss���>


accuracy_1�z?�}�D]       a[��	�'�+_��A�*O

prediction_lossq=
?

reg_loss2xw<


total_lossR?


accuracy_1��>��]       a[��	�I�+_��A�*O

prediction_lossq=
?

reg_loss(xw<


total_lossR?


accuracy_1��>D]��]       a[��	Eg�+_��A�*O

prediction_loss�?

reg_lossxw<


total_loss��?


accuracy_1���>��J]       a[��	Q��+_��A�*O

prediction_loss�G�>

reg_lossxw<


total_losso�>


accuracy_1)\?J:?]       a[��	ۣ�+_��A�*O

prediction_loss���>

reg_loss	xw<


total_loss���>


accuracy_1��?��ې]       a[��	P��+_��A�*O

prediction_loss   ?

reg_loss xw<


total_loss��?


accuracy_1   ?��.]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss�ww<


total_loss��?


accuracy_1   ?� ��]       a[��	���+_��A�*O

prediction_loss��?

reg_loss�ww<


total_losszw?


accuracy_1���>~���]       a[��	��+_��A�*O

prediction_loss=
�>

reg_loss�ww<


total_loss���>


accuracy_1�z?��]       a[��	�=�+_��A�*O

prediction_loss{�>

reg_loss�ww<


total_loss:е>


accuracy_1��(?��(]       a[��	F\�+_��A�*O

prediction_loss�?

reg_loss�ww<


total_loss��?


accuracy_1���>C]]       a[��	�{�+_��A�*O

prediction_loss�z?

reg_loss�ww<


total_loss�X?


accuracy_1=
�>=C��]       a[��	��+_��A�*O

prediction_loss���>

reg_loss�ww<


total_loss���>


accuracy_1��?�|2a]       a[��	F��+_��A�*O

prediction_loss\��>

reg_loss�ww<


total_lossK�>


accuracy_1R�?c�C]       a[��	t��+_��A�*O

prediction_loss��>

reg_loss�ww<


total_loss�@�>


accuracy_1q=
?mz9�]       a[��	B��+_��A�*O

prediction_loss���>

reg_loss�ww<


total_lossWU�>


accuracy_1333?�q��]       a[��	�
�+_��A�*O

prediction_loss���>

reg_loss�ww<


total_lossK~�>


accuracy_1�?>}=]       a[��	(+�+_��A�*O

prediction_loss\��>

reg_loss�ww<


total_lossK�>


accuracy_1R�?��]       a[��	^H�+_��A�*O

prediction_loss�?

reg_lossyww<


total_loss��?


accuracy_1���>��(]       a[��	�g�+_��A�*O

prediction_loss�z?

reg_lossoww<


total_loss�X?


accuracy_1=
�>gl��]       a[��	Ɖ�+_��A�*O

prediction_loss   ?

reg_losseww<


total_loss��?


accuracy_1   ??ފ]       a[��	U��+_��A�*O

prediction_loss   ?

reg_lossZww<


total_loss��?


accuracy_1   ?ac=]       a[��	���+_��A�*O

prediction_loss{�>

reg_lossNww<


total_loss5е>


accuracy_1��(?P��]       a[��	 ��+_��A�*O

prediction_lossq=
?

reg_lossDww<


total_lossN?


accuracy_1��>H��]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss;ww<


total_loss��?


accuracy_1   ?.ίX]       a[��	�$�+_��A�*O

prediction_loss���>

reg_loss1ww<


total_lossI~�>


accuracy_1�?.+E]       a[��	�>�+_��A�*O

prediction_loss
�#?

reg_loss%ww<


total_loss�'?


accuracy_1�Q�>�n�}]       a[��	`Y�+_��A�*O

prediction_loss���>

reg_lossww<


total_lossH~�>


accuracy_1�?>K�!]       a[��	`r�+_��A�*O

prediction_loss=
�>

reg_lossww<


total_loss���>


accuracy_1�z?@��]       a[��	?��+_��A�*O

prediction_loss=
�>

reg_lossww<


total_loss���>


accuracy_1�z?��x]       a[��	+��+_��A�*O

prediction_loss�?

reg_loss�vw<


total_loss��?


accuracy_1���>��]       a[��	��+_��A�*O

prediction_lossq=
?

reg_loss�vw<


total_lossM?


accuracy_1��>�3&�]       a[��	r��+_��A�*O

prediction_loss�G�>

reg_loss�vw<


total_losse�>


accuracy_1)\?���]       a[��	���+_��A�*O

prediction_loss�G�>

reg_loss�vw<


total_losse�>


accuracy_1)\? HΎ]       a[��	��+_��A�*O

prediction_lossq=
?

reg_loss�vw<


total_lossL?


accuracy_1��>�Zk�]       a[��	'3�+_��A�*O

prediction_loss�?

reg_loss�vw<


total_loss��?


accuracy_1���>Kl�]       a[��	M�+_��A�*O

prediction_loss\��>

reg_loss�vw<


total_lossK�>


accuracy_1R�?Ha�]       a[��	�f�+_��A�*O

prediction_loss�z?

reg_loss�vw<


total_loss�X?


accuracy_1=
�>���]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss�vw<


total_loss��?


accuracy_1   ?���]       a[��	��+_��A�*O

prediction_loss�?

reg_loss�vw<


total_loss��?


accuracy_1���>Ť��]       a[��	g��+_��A�*O

prediction_loss���>

reg_loss�vw<


total_loss���>


accuracy_1��?�mr(]       a[��	���+_��A�*O

prediction_loss�G�>

reg_loss�vw<


total_lossb�>


accuracy_1)\?û��]       a[��	!��+_��A�*O

prediction_loss��>

reg_loss�vw<


total_loss�@�>


accuracy_1q=
?�%]       a[��	��+_��A�*O

prediction_loss=
�>

reg_losswvw<


total_loss���>


accuracy_1�z? ]��]       a[��	��+_��A�*O

prediction_lossq=
?

reg_lossovw<


total_lossK?


accuracy_1��>�m]       a[��	�2�+_��A�*O

prediction_loss���>

reg_lossdvw<


total_lossB~�>


accuracy_1�?h�?�]       a[��	�L�+_��A�*O

prediction_loss���>

reg_lossYvw<


total_lossB~�>


accuracy_1�?Q��a]       a[��	sg�+_��A�*O

prediction_loss���>

reg_lossNvw<


total_lossA~�>


accuracy_1�? �/�]       a[��	���+_��A�*O

prediction_loss�?

reg_lossEvw<


total_loss��?


accuracy_1���>ה2.]       a[��	N��+_��A�*O

prediction_loss   ?

reg_loss:vw<


total_loss��?


accuracy_1   ? ���]       a[��	ö�+_��A�*O

prediction_loss   ?

reg_loss0vw<


total_loss��?


accuracy_1   ?z�p]       a[��	N��+_��A�*O

prediction_loss�G�>

reg_loss%vw<


total_loss_�>


accuracy_1)\?��D�]       a[��	���+_��A�*O

prediction_loss�?

reg_lossvw<


total_loss��?


accuracy_1���>�i�]       a[��	q�+_��A�*O

prediction_loss�G�>

reg_lossvw<


total_loss^�>


accuracy_1)\?�9ӛ]       a[��	&�+_��A�*O

prediction_loss���>

reg_lossvw<


total_lossJU�>


accuracy_1333?�5�%]       a[��	4�+_��A�*O

prediction_loss��?

reg_loss�uw<


total_lossrw?


accuracy_1���>��fP]       a[��	S�+_��A�*O

prediction_loss���>

reg_loss�uw<


total_loss?~�>


accuracy_1�?^mX]       a[��	�k�+_��A�*O

prediction_lossq=
?

reg_loss�uw<


total_lossI?


accuracy_1��>�7i�]       a[��	"��+_��A�*O

prediction_loss=
�>

reg_loss�uw<


total_loss���>


accuracy_1�z?��ƪ]       a[��	&��+_��A�*O

prediction_lossq=
?

reg_loss�uw<


total_lossH?


accuracy_1��>PT�]       a[��	c��+_��A�*O

prediction_loss\��>

reg_loss�uw<


total_loss
K�>


accuracy_1R�?���m]       a[��	A��+_��A�*O

prediction_loss���>

reg_loss�uw<


total_loss=~�>


accuracy_1�?�A��]       a[��	��+_��A�*O

prediction_loss���>

reg_loss�uw<


total_loss=~�>


accuracy_1�?�c]       a[��	!#�+_��A�*O

prediction_loss�G�>

reg_loss�uw<


total_loss[�>


accuracy_1)\?3��]       a[��	nM�+_��A�*O

prediction_loss�?

reg_loss�uw<


total_loss��?


accuracy_1���>;��J]       a[��	tz�+_��A�*O

prediction_loss)\?

reg_loss�uw<


total_loss�9?


accuracy_1�G�>�E�]       a[��	ߥ�+_��A�*O

prediction_loss���>

reg_loss�uw<


total_loss;~�>


accuracy_1�?�T�d]       a[��	���+_��A�*O

prediction_loss   ?

reg_loss�uw<


total_loss��?


accuracy_1   ?�&�]       a[��	R��+_��A�*O

prediction_loss�z?

reg_losswuw<


total_loss�X?


accuracy_1=
�>]$\]       a[��	f�+_��A�*O

prediction_loss=
�>

reg_losskuw<


total_loss���>


accuracy_1�z?��8W]       a[��	1�+_��A�*O

prediction_loss   ?

reg_lossauw<


total_loss��?


accuracy_1   ?fo�Z]       a[��	�N�+_��A�*O

prediction_lossq=
?

reg_lossYuw<


total_lossF?


accuracy_1��>j1]       a[��	bj�+_��A�*O

prediction_loss333?

reg_lossLuw<


total_loss7?


accuracy_1���>S7�i]       a[��	'��+_��A�*O

prediction_loss)\?

reg_lossCuw<


total_loss�9?


accuracy_1�G�>\�]       a[��	T��+_��A�*O

prediction_loss�?

reg_loss8uw<


total_loss��?


accuracy_1���>��]       a[��	�=�+_��A�*O

prediction_loss)\?

reg_loss$uw<


total_loss�9?


accuracy_1�G�>���]       a[��	�j�+_��A�*O

prediction_loss)\?

reg_lossuw<


total_loss�9?


accuracy_1�G�>���8]       a[��	���+_��A�*O

prediction_loss=
�>

reg_lossuw<


total_loss���>


accuracy_1�z?4�;�]       a[��	8��+_��A�*O

prediction_loss�?

reg_lossuw<


total_loss��?


accuracy_1���>���]       a[��	r�+_��A�*O

prediction_loss��>

reg_loss�tw<


total_loss�@�>


accuracy_1q=
?���]       a[��	�f�+_��A�*O

prediction_lossR�?

reg_loss�tw<


total_loss&�"?


accuracy_1\��>�)8@]       a[��	O��+_��A�*O

prediction_loss=
�>

reg_loss�tw<


total_loss���>


accuracy_1�z?U:�]       a[��	f��+_��A�*O

prediction_lossq=
?

reg_loss�tw<


total_lossD?


accuracy_1��>��]       a[��	���+_��A�*O

prediction_loss���>

reg_loss�tw<


total_losst��>


accuracy_1��?��]       a[��	U��+_��A�*O

prediction_loss)\?

reg_loss�tw<


total_loss�9?


accuracy_1�G�>`��q]       a[��	-��+_��A�*O

prediction_loss\��>

reg_loss�tw<


total_lossK�>


accuracy_1R�?i�3N]       a[��	���+_��A�*O

prediction_loss�z?

reg_loss�tw<


total_loss�X?


accuracy_1=
�>���p]       a[��	�,�+_��A�*O

prediction_loss�?

reg_loss�tw<


total_loss��?


accuracy_1���>;�s�]       a[��	��+_��A�*O

prediction_loss���>

reg_loss�tw<


total_loss4~�>


accuracy_1�?�w?�]       a[��	�+_��A�*O

prediction_loss���>

reg_loss�tw<


total_loss4~�>


accuracy_1�?��yT]       a[��	�B�+_��A�*O

prediction_loss�G�>

reg_loss�tw<


total_lossR�>


accuracy_1)\?�e]       a[��	�q�+_��A�*O

prediction_loss�Q8?

reg_loss~tw<


total_loss�/<?


accuracy_1)\�>�l�]       a[��	Ș�+_��A�*O

prediction_loss�G�>

reg_lossutw<


total_lossR�>


accuracy_1)\?�Jm]       a[��	���+_��A�*O

prediction_loss���>

reg_lossltw<


total_loss=U�>


accuracy_1333?j�u]       a[��	D�+_��A�*O

prediction_loss���>

reg_loss`tw<


total_loss2~�>


accuracy_1�?d��]       a[��	^M�+_��A�*O

prediction_lossq=
?

reg_lossVtw<


total_lossB?


accuracy_1��>���]       a[��	]n�+_��A�*O

prediction_loss�z?

reg_lossLtw<


total_loss�X?


accuracy_1=
�>-~\�]       a[��	g��+_��A�*O

prediction_loss=
�>

reg_loss@tw<


total_loss���>


accuracy_1�z?�p]       a[��	O��+_��A�*O

prediction_loss   ?

reg_loss7tw<


total_loss��?


accuracy_1   ?��1]       a[��	���+_��A�*O

prediction_lossq=
?

reg_loss,tw<


total_lossB?


accuracy_1��>P.�]       a[��	�*�+_��A�*O

prediction_loss�?

reg_loss tw<


total_loss��?


accuracy_1���>x�ְ]       a[��	Nb�+_��A�*O

prediction_loss\��>

reg_losstw<


total_loss�J�>


accuracy_1R�?�~M�]       a[��	X��+_��A�*O

prediction_loss��?

reg_losstw<


total_lossjw?


accuracy_1���>�L"]       a[��	���+_��A�*O

prediction_loss�?

reg_losstw<


total_loss��?


accuracy_1���>�f]       a[��	��+_��A�*O

prediction_loss=
�>

reg_loss�sw<


total_loss���>


accuracy_1�z?�bS�]       a[��	��+_��A�*O

prediction_loss   ?

reg_loss�sw<


total_loss��?


accuracy_1   ?Mq�]       a[��	]1�+_��A�*O

prediction_loss���>

reg_loss�sw<


total_loss.~�>


accuracy_1�?�j�]       a[��	Lq�+_��A�*O

prediction_lossR�?

reg_loss�sw<


total_loss!�"?


accuracy_1\��>��:]]       a[��	��+_��A�*O

prediction_loss�?

reg_loss�sw<


total_loss��?


accuracy_1���>" e�]       a[��	ͬ�+_��A�*O

prediction_loss)\?

reg_loss�sw<


total_loss�9?


accuracy_1�G�>�Oc]       a[��	���+_��A�*O

prediction_loss�z?

reg_loss�sw<


total_loss�X?


accuracy_1=
�>�
O�]       a[��	X��+_��A�*O

prediction_loss�G�>

reg_loss�sw<


total_lossL�>


accuracy_1)\?T$+G]       a[��	" �+_��A�*O

prediction_loss   ?

reg_loss�sw<


total_loss��?


accuracy_1   ?6\�,]       a[��	q�+_��A�*O

prediction_loss�?

reg_loss�sw<


total_loss��?


accuracy_1���>h��v]       a[��	}<�+_��A�*O

prediction_loss\��>

reg_loss�sw<


total_loss�J�>


accuracy_1R�?�w�	]       a[��	>Z�+_��A�*O

prediction_loss�G�>

reg_loss�sw<


total_lossJ�>


accuracy_1)\?�0�]       a[��	�y�+_��A�*O

prediction_loss���>

reg_loss|sw<


total_lossi��>


accuracy_1��?;FT�]       a[��	���+_��A�*O

prediction_loss   ?

reg_lossqsw<


total_loss��?


accuracy_1   ?�]       a[��	-��+_��A�*O

prediction_loss�z?

reg_lossdsw<


total_loss�X?


accuracy_1=
�>�V�]       a[��	���+_��A�*O

prediction_loss=
�>

reg_loss\sw<


total_loss���>


accuracy_1�z?Ef��]       a[��	p��+_��A�*O

prediction_loss���>

reg_lossQsw<


total_lossh��>


accuracy_1��?���]       a[��	J
�+_��A�*O

prediction_loss�?

reg_lossGsw<


total_loss��?


accuracy_1���>��Z]       a[��	�%�+_��A�*O

prediction_loss���>

reg_loss<sw<


total_loss)~�>


accuracy_1�?.�+]       a[��	1A�+_��A�*O

prediction_loss�?

reg_loss3sw<


total_loss��?


accuracy_1���> �.5]       a[��	�X�+_��A�*O

prediction_loss   ?

reg_loss(sw<


total_loss��?


accuracy_1   ?pI�?]       a[��	�q�+_��A�*O

prediction_loss��>

reg_losssw<


total_loss�@�>


accuracy_1q=
?�K�d]       a[��	 ��+_��A�*O

prediction_loss�z?

reg_losssw<


total_loss�X?


accuracy_1=
�>9��]       a[��	r��+_��A�*O

prediction_loss��>

reg_losssw<


total_loss�@�>


accuracy_1q=
?��]       a[��	���+_��A�*O

prediction_loss�?

reg_loss sw<


total_loss��?


accuracy_1���>�1g�]       a[��	w��+_��A�*O

prediction_loss�z?

reg_loss�rw<


total_loss�X?


accuracy_1=
�>�KV]       a[��	���+_��A�*O

prediction_loss��>

reg_loss�rw<


total_loss�@�>


accuracy_1q=
?gY+�]       a[��	��+_��A�*O

prediction_loss�?

reg_loss�rw<


total_loss��?


accuracy_1���>�_a�]       a[��	-�+_��A�*O

prediction_loss\��>

reg_loss�rw<


total_loss�J�>


accuracy_1R�?�Fר]       a[��	�J�+_��A�*O

prediction_loss���>

reg_loss�rw<


total_loss%~�>


accuracy_1�?�ޠ.]       a[��	�f�+_��A�*O

prediction_lossq=
?

reg_loss�rw<


total_loss<?


accuracy_1��>�ZZ�]       a[��	'��+_��A�*O

prediction_lossq=
?

reg_loss�rw<


total_loss<?


accuracy_1��>���]       a[��	R��+_��A�*O

prediction_loss)\?

reg_loss�rw<


total_loss�9?


accuracy_1�G�>[WS]       a[��	���+_��A�*O

prediction_loss)\?

reg_loss�rw<


total_loss�9?


accuracy_1�G�>םj]       a[��	J��+_��A�*O

prediction_loss   ?

reg_loss�rw<


total_loss��?


accuracy_1   ?pB�!]       a[��	F��+_��A�*O

prediction_loss��>

reg_loss�rw<


total_loss�@�>


accuracy_1q=
?�`�	]       a[��	S�+_��A�*O

prediction_loss\��>

reg_loss�rw<


total_loss�J�>


accuracy_1R�?��6X]       a[��	�!�+_��A�*O

prediction_loss��>

reg_losswrw<


total_loss�@�>


accuracy_1q=
?�Lm]       a[��	VH�+_��A�*O

prediction_loss�?

reg_lossmrw<


total_loss��?


accuracy_1���>�'j�]       a[��	Vc�+_��A�*O

prediction_loss)\?

reg_lossbrw<


total_loss�9?


accuracy_1�G�>��֤]       a[��	x}�+_��A�*O

prediction_lossq=
?

reg_lossXrw<


total_loss:?


accuracy_1��>~o�Q]       a[��	ȗ�+_��A�*O

prediction_loss   ?

reg_lossPrw<


total_loss��?


accuracy_1   ?1��+]       a[��	u��+_��A�*O

prediction_loss�z?

reg_lossErw<


total_loss�X?


accuracy_1=
�>k#�]       a[��	���+_��A�*O

prediction_loss��>

reg_loss9rw<


total_loss�@�>


accuracy_1q=
?��&�]       a[��	i��+_��A�*O

prediction_loss=
�>

reg_loss0rw<


total_loss���>


accuracy_1�z?C�Y�]       a[��	a��+_��A�*O

prediction_loss���>

reg_loss#rw<


total_loss^��>


accuracy_1��?���]       a[��	��+_��A�*O

prediction_loss)\?

reg_lossrw<


total_loss�9?


accuracy_1�G�>�>]       a[��	v5�+_��A�*O

prediction_loss��?

reg_lossrw<


total_lossbw?


accuracy_1���>6x[�]       a[��	�W�+_��A�*O

prediction_loss
�#?

reg_lossrw<


total_lossҴ'?


accuracy_1�Q�>7�@]       a[��	py�+_��A�*O

prediction_loss�z?

reg_loss�qw<


total_loss�X?


accuracy_1=
�>V�PA]       a[��	��+_��A�*O

prediction_lossq=
?

reg_loss�qw<


total_loss9?


accuracy_1��>�;��]       a[��	A��+_��A�*O

prediction_loss=
�>

reg_loss�qw<


total_loss���>


accuracy_1�z?�n[1]       a[��	5��+_��A�*O

prediction_loss=
�>

reg_loss�qw<


total_loss���>


accuracy_1�z?��]       a[��	���+_��A�*O

prediction_loss��>

reg_loss�qw<


total_loss�@�>


accuracy_1q=
?���1]       a[��	|
�+_��A�*O

prediction_loss�?

reg_loss�qw<


total_loss�?


accuracy_1���>׊c�]       a[��	�"�+_��A�*O

prediction_loss=
�>

reg_loss�qw<


total_loss���>


accuracy_1�z?pep]       a[��	�>�+_��A�*O

prediction_loss=
�>

reg_loss�qw<


total_loss���>


accuracy_1�z?�UC]       a[��	6W�+_��A�*O

prediction_lossq=
?

reg_loss�qw<


total_loss8?


accuracy_1��>�<�]       a[��	o�+_��A�*O

prediction_loss)\?

reg_loss�qw<


total_loss�9?


accuracy_1�G�>�^_�]       a[��	M��+_��A�*O

prediction_loss�?

reg_loss�qw<


total_loss~�?


accuracy_1���>�'x�]       a[��	$��+_��A�*O

prediction_lossq=
?

reg_loss�qw<


total_loss7?


accuracy_1��>2D�z]       a[��	��+_��A�*O

prediction_lossq=
?

reg_loss�qw<


total_loss7?


accuracy_1��>Q"�A]       a[��	���+_��A�*O

prediction_loss��>

reg_lossyqw<


total_loss�@�>


accuracy_1q=
?��]       a[��	���+_��A�*O

prediction_loss=
�>

reg_losspqw<


total_loss���>


accuracy_1�z?!�i�]       a[��	���+_��A�*O

prediction_loss��>

reg_lossdqw<


total_loss�@�>


accuracy_1q=
?A'�{]       a[��	��+_��A�*O

prediction_loss�?

reg_loss[qw<


total_loss}�?


accuracy_1���>=�]       a[��	�4�+_��A�*O

prediction_loss�Q�>

reg_lossQqw<


total_lossw�>


accuracy_1
�#?��|5]       a[��	rM�+_��A�*O

prediction_loss���>

reg_lossHqw<


total_loss~�>


accuracy_1�?��~]       a[��	ke�+_��A�*O

prediction_loss��>

reg_loss;qw<


total_loss�@�>


accuracy_1q=
?�]       a[��	k��+_��A�*O

prediction_loss\��>

reg_loss1qw<


total_loss�J�>


accuracy_1R�?�t|�]       a[��	���+_��A�*O

prediction_loss�G�>

reg_loss(qw<


total_loss7�>


accuracy_1)\?%���]       a[��	���+_��A�*O

prediction_loss��>

reg_lossqw<


total_loss�@�>


accuracy_1q=
?�PI]       a[��	�+_��A�*O

prediction_loss��>

reg_lossqw<


total_loss�@�>


accuracy_1q=
?�T-�]       a[��	�+_��A�*O

prediction_loss���>

reg_loss�pw<


total_loss~�>


accuracy_1�?�XY]       a[��	�1�+_��A�*O

prediction_lossq=
?

reg_loss�pw<


total_loss5?


accuracy_1��>��U]       a[��	�X�+_��A�*O

prediction_loss���>

reg_loss�pw<


total_lossT��>


accuracy_1��?��]       a[��	uv�+_��A�*O

prediction_lossq=
?

reg_loss�pw<


total_loss4?


accuracy_1��>�+=]       a[��	��+_��A�*O

prediction_loss�?

reg_loss�pw<


total_loss{�?


accuracy_1���>��S2]       a[��	���+_��A�*O

prediction_loss�G�>

reg_loss�pw<


total_loss4�>


accuracy_1)\?��xe]       a[��	���+_��A�*O

prediction_loss�z?

reg_loss�pw<


total_loss�X?


accuracy_1=
�>7-�]       a[��	��+_��A�*O

prediction_loss�z?

reg_loss�pw<


total_loss�X?


accuracy_1=
�>���]       a[��	�-�+_��A�*O

prediction_loss=
�>

reg_loss�pw<


total_loss���>


accuracy_1�z?qg��]       a[��	$|�+_��A�*O

prediction_loss�Q�>

reg_loss�pw<


total_lossq�>


accuracy_1
�#?��e]       a[��	v��+_��A�*O

prediction_loss���>

reg_loss�pw<


total_lossR��>


accuracy_1��?�ކ�]       a[��	a��+_��A�*O

prediction_loss���>

reg_loss�pw<


total_loss~�>


accuracy_1�?DU+�]       a[��	C�+_��A�*O

prediction_lossq=
?

reg_loss�pw<


total_loss3?


accuracy_1��>�/J�]       a[��	�d�+_��A�*O

prediction_loss)\?

reg_lossupw<


total_loss�9?


accuracy_1�G�>'[�]       a[��	��+_��A�*O

prediction_loss\��>

reg_losslpw<


total_loss�J�>


accuracy_1R�?&��]       a[��	��+_��A�*O

prediction_loss)\?

reg_losscpw<


total_loss�9?


accuracy_1�G�>b�D]       a[��	O��+_��A�*O

prediction_loss��?

reg_lossYpw<


total_loss[w?


accuracy_1���>���`]       a[��	)�+_��A�*O

prediction_loss�z?

reg_lossOpw<


total_loss�X?


accuracy_1=
�>��m]       a[��	�~�+_��A�*O

prediction_loss�?

reg_lossEpw<


total_lossy�?


accuracy_1���>SN��]       a[��	��+_��A�*O

prediction_loss�?

reg_loss9pw<


total_lossy�?


accuracy_1���>��]       a[��	���+_��A�*O

prediction_loss   ?

reg_loss0pw<


total_loss��?


accuracy_1   ?3���]       a[��	t ,_��A�*O

prediction_loss�?

reg_loss&pw<


total_lossy�?


accuracy_1���>�`��]       a[��	�F ,_��A�*O

prediction_loss�G�>

reg_losspw<


total_loss/�>


accuracy_1)\?1d ]       a[��		q ,_��A�*O

prediction_loss��>

reg_losspw<


total_loss�@�>


accuracy_1q=
?~-�o]       a[��	I� ,_��A�*O

prediction_loss=
�>

reg_losspw<


total_loss���>


accuracy_1�z?o��Q]       a[��	�� ,_��A�*O

prediction_lossq=
?

reg_loss�ow<


total_loss1?


accuracy_1��>O�W�]       a[��	/� ,_��A�*O

prediction_loss   ?

reg_loss�ow<


total_loss��?


accuracy_1   ?�.�G]       a[��	�@,_��A�*O

prediction_loss)\?

reg_loss�ow<


total_loss�9?


accuracy_1�G�>$
�]       a[��	�o,_��A�*O

prediction_loss)\?

reg_loss�ow<


total_loss�9?


accuracy_1�G�>? 3�]       a[��	
�,_��A�*O

prediction_loss)\?

reg_loss�ow<


total_loss�9?


accuracy_1�G�>��]       a[��	��,_��A�*O

prediction_loss���>

reg_loss�ow<


total_loss~�>


accuracy_1�?�u�]       a[��	�	,_��A�*O

prediction_loss   ?

reg_loss�ow<


total_loss��?


accuracy_1   ?���]       a[��	�G,_��A�*O

prediction_loss   ?

reg_loss�ow<


total_loss��?


accuracy_1   ?�3hL]       a[��	�p,_��A�*O

prediction_loss   ?

reg_loss�ow<


total_loss��?


accuracy_1   ?U0��]       a[��	]�,_��A�*O

prediction_lossq=
?

reg_loss�ow<


total_loss/?


accuracy_1��>�i�}]       a[��	|�,_��A�*O

prediction_loss�?

reg_loss�ow<


total_lossv�?


accuracy_1���>�X�']       a[��	,_��A�*O

prediction_loss��>

reg_loss�ow<


total_loss�@�>


accuracy_1q=
?��]       a[��	*8,_��A�*O

prediction_loss�z?

reg_loss�ow<


total_loss�X?


accuracy_1=
�>)H�]       a[��	�[,_��A�*O

prediction_loss���>

reg_lossvow<


total_loss~�>


accuracy_1�?��n]]       a[��	O�,_��A�*O

prediction_lossq=
?

reg_losskow<


total_loss/?


accuracy_1��>�VO�]       a[��	�,_��A�*O

prediction_lossR�?

reg_lossaow<


total_loss�"?


accuracy_1\��>��(�]       a[��	,_��A�*O

prediction_loss���>

reg_lossWow<


total_loss
~�>


accuracy_1�?�ߍ]       a[��	�D,_��A�*O

prediction_loss���>

reg_lossKow<


total_loss	~�>


accuracy_1�?2tpk]       a[��	bi,_��A�*O

prediction_loss   ?

reg_lossAow<


total_loss��?


accuracy_1   ?�8ӱ]       a[��	v�,_��A�*O

prediction_loss���>

reg_loss6ow<


total_lossG��>


accuracy_1��?�ܥ]       a[��	%�,_��A�*O

prediction_loss   ?

reg_loss,ow<


total_loss��?


accuracy_1   ? ;5�]       a[��	�,_��A�*O

prediction_loss��?

reg_loss ow<


total_lossVw?


accuracy_1���>6յv]       a[��	�,_��A�*O

prediction_loss���>

reg_lossow<


total_loss~�>


accuracy_1�?B�x]       a[��	�O,_��A�*O

prediction_loss   ?

reg_lossow<


total_loss��?


accuracy_1   ?px�]       a[��	_z,_��A�*O

prediction_loss)\?

reg_lossow<


total_loss�9?


accuracy_1�G�>%D%�]       a[��	ʣ,_��A�*O

prediction_loss�?

reg_loss�nw<


total_losst�?


accuracy_1���>�9f�]       a[��	P�,_��A�*O

prediction_loss)\?

reg_loss�nw<


total_loss�9?


accuracy_1�G�>�P_y]       a[��	Z�,_��A�*O

prediction_loss�?

reg_loss�nw<


total_losst�?


accuracy_1���>���&]       a[��	a7,_��A�*O

prediction_loss�?

reg_loss�nw<


total_losss�?


accuracy_1���>���]       a[��	�d,_��A�*O

prediction_lossq=
?

reg_loss�nw<


total_loss,?


accuracy_1��>��w]       a[��	T�,_��A�*O

prediction_loss��>

reg_loss�nw<


total_loss�@�>


accuracy_1q=
?�/�]       a[��	�,_��A�*O

prediction_loss���>

reg_loss�nw<


total_loss~�>


accuracy_1�?����]       a[��	&�,_��A�*O

prediction_lossq=
?

reg_loss�nw<


total_loss,?


accuracy_1��>y_V�]       a[��	L,_��A�*O

prediction_loss��>

reg_loss�nw<


total_loss�@�>


accuracy_1q=
?��M]       a[��	U,_��A�*O

prediction_loss�G�>

reg_loss�nw<


total_loss#�>


accuracy_1)\?�]{�]       a[��	^�,_��A�*O

prediction_lossq=
?

reg_loss�nw<


total_loss+?


accuracy_1��>]E��]       a[��	�,_��A�*O

prediction_loss)\?

reg_loss�nw<


total_loss�9?


accuracy_1�G�>�j�?]       a[��	��,_��A�*O

prediction_loss)\?

reg_loss~nw<


total_loss�9?


accuracy_1�G�>L�!]       a[��	W,_��A�*O

prediction_loss��>

reg_losssnw<


total_loss�@�>


accuracy_1q=
?��5]       a[��	B,_��A�*O

prediction_loss���>

reg_lossinw<


total_loss~�>


accuracy_1�?.�]       a[��	�s,_��A�*O

prediction_loss���>

reg_loss^nw<


total_loss~�>


accuracy_1�?�=�]       a[��	�,_��A�*O

prediction_loss\��>

reg_lossTnw<


total_loss�J�>


accuracy_1R�?��p	]       a[��	��,_��A�*O

prediction_loss��>

reg_lossLnw<


total_loss�@�>


accuracy_1q=
?���j]       a[��	u�,_��A�*O

prediction_loss���>

reg_lossBnw<


total_loss~�>


accuracy_1�?�^�3]       a[��	� 	,_��A�*O

prediction_loss���>

reg_loss5nw<


total_loss~�>


accuracy_1�?ڭ%�]       a[��	�!	,_��A�*O

prediction_loss��>

reg_loss+nw<


total_loss�@�>


accuracy_1q=
?L�O<]       a[��	�B	,_��A�*O

prediction_loss=
�>

reg_loss!nw<


total_loss���>


accuracy_1�z?gL�]       a[��	9_	,_��A�*O

prediction_loss���>

reg_lossnw<


total_loss ~�>


accuracy_1�?4pH]       a[��	M�	,_��A�*O

prediction_lossq=
?

reg_lossnw<


total_loss)?


accuracy_1��>��g�]       a[��	ץ	,_��A�*O

prediction_loss���>

reg_lossnw<


total_loss�}�>


accuracy_1�?��)�]       a[��	��	,_��A�*O

prediction_loss���>

reg_loss�mw<


total_loss�}�>


accuracy_1�?��C�]       a[��	��	,_��A�*O

prediction_lossq=
?

reg_loss�mw<


total_loss)?


accuracy_1��>�6��]       a[��	�
,_��A�*O

prediction_loss   ?

reg_loss�mw<


total_loss��?


accuracy_1   ?�Z#M]       a[��	�C
,_��A�*O

prediction_loss   ?

reg_loss�mw<


total_loss��?


accuracy_1   ?�o�]       a[��	bh
,_��A�*O

prediction_lossR�?

reg_loss�mw<


total_loss	�"?


accuracy_1\��>PQ�(]       a[��	��
,_��A�*O

prediction_loss   ?

reg_loss�mw<


total_loss��?


accuracy_1   ?G�M/]       a[��	G�
,_��A�*O

prediction_loss��>

reg_loss�mw<


total_loss�@�>


accuracy_1q=
?�Q��]       a[��	�,_��A�*O

prediction_loss���>

reg_loss�mw<


total_loss�}�>


accuracy_1�?���2]       a[��	\>,_��A�*O

prediction_loss=
�>

reg_loss�mw<


total_loss���>


accuracy_1�z?tx�U]       a[��	�a,_��A�*O

prediction_loss���>

reg_loss�mw<


total_loss:��>


accuracy_1��?�Ŝ']       a[��	#�,_��A�*O

prediction_loss�z?

reg_loss�mw<


total_loss�X?


accuracy_1=
�>�_�]       a[��	��,_��A�*O

prediction_loss���>

reg_loss�mw<


total_loss�}�>


accuracy_1�?n\��]       a[��	
,_��A�*O

prediction_loss   ?

reg_lossymw<


total_loss��?


accuracy_1   ?�r]       a[��	�6,_��A�*O

prediction_lossq=
?

reg_lossqmw<


total_loss'?


accuracy_1��>��n�]       a[��	�X,_��A�*O

prediction_loss���>

reg_lossfmw<


total_loss�}�>


accuracy_1�?�r�	]       a[��	�,_��A�*O

prediction_loss=
�>

reg_loss\mw<


total_loss���>


accuracy_1�z?f��l]       a[��	�,_��A�*O

prediction_lossq=
?

reg_lossQmw<


total_loss&?


accuracy_1��>���]       a[��	M,_��A�*O

prediction_loss{�>

reg_lossEmw<


total_loss�ϵ>


accuracy_1��(?��]       a[��	�.,_��A�*O

prediction_loss{.?

reg_loss<mw<


total_loss0�1?


accuracy_1
ף>t���]       a[��	>^,_��A�*O

prediction_loss�?

reg_loss3mw<


total_lossm�?


accuracy_1���>��_%]       a[��	q�,_��A�*O

prediction_loss   ?

reg_loss(mw<


total_loss��?


accuracy_1   ?{g�0]       a[��	��,_��A�*O

prediction_loss   ?

reg_lossmw<


total_loss��?


accuracy_1   ?YR]       a[��	,,_��A�*O

prediction_loss=
�>

reg_lossmw<


total_loss���>


accuracy_1�z?��Ig]       a[��	߇,_��A�*O

prediction_loss���>

reg_loss�lw<


total_loss�}�>


accuracy_1�?Б�]       a[��	��,_��A�*O

prediction_loss�G�>

reg_loss�lw<


total_loss�>


accuracy_1)\?1��]       a[��	��,_��A�*O

prediction_loss)\?

reg_loss�lw<


total_loss�9?


accuracy_1�G�>x��]       a[��	,.,_��A�*O

prediction_loss=
�>

reg_loss�lw<


total_loss���>


accuracy_1�z?��o�]       a[��	�V,_��A�*O

prediction_lossR�?

reg_loss�lw<


total_loss�"?


accuracy_1\��>!s�]       a[��	,�,_��A�*O

prediction_loss���>

reg_loss�lw<


total_loss�}�>


accuracy_1�?���N]       a[��	ު,_��A�*O

prediction_loss)\?

reg_loss�lw<


total_loss�9?


accuracy_1�G�>�b&]       a[��	��,_��A�*O

prediction_loss
ף>

reg_loss�lw<


total_lossp��>


accuracy_1{.?�j�]       a[��	1�,_��A�*O

prediction_loss   ?

reg_loss�lw<


total_loss��?


accuracy_1   ?�\�]       a[��	�B,_��A�*O

prediction_loss���>

reg_loss�lw<


total_loss�}�>


accuracy_1�?z��]       a[��	�z,_��A�*O

prediction_loss�?

reg_loss�lw<


total_lossj�?


accuracy_1���>o�I]       a[��	��,_��A�*O

prediction_loss   ?

reg_loss�lw<


total_loss��?


accuracy_1   ?��<�]       a[��	��,_��A�*O

prediction_loss=
�>

reg_loss�lw<


total_loss���>


accuracy_1�z?� �]       a[��	S,_��A�*O

prediction_loss{�>

reg_losswlw<


total_loss�ϵ>


accuracy_1��(?�9y�]       a[��	�h,_��A�*O

prediction_loss�z?

reg_lossnlw<


total_loss�X?


accuracy_1=
�>�2��]       a[��	t�,_��A�*O

prediction_loss���>

reg_lossblw<


total_loss�}�>


accuracy_1�?�c�"]       a[��	p�,_��A�*O

prediction_loss���>

reg_lossZlw<


total_loss�}�>


accuracy_1�?�5�]       a[��	
�,_��A�*O

prediction_loss���>

reg_lossOlw<


total_loss�}�>


accuracy_1�?�͖R]       a[��	G,_��A�*O

prediction_loss���>

reg_lossElw<


total_loss�}�>


accuracy_1�?��q]       a[��	�Z,_��A�*O

prediction_loss=
�>

reg_loss;lw<


total_loss���>


accuracy_1�z?��_�]       a[��	9�,_��A�*O

prediction_loss   ?

reg_loss2lw<


total_loss��?


accuracy_1   ?�C�]       a[��	��,_��A�*O

prediction_loss   ?

reg_loss'lw<


total_loss��?


accuracy_1   ?��]       a[��	�,_��A�*O

prediction_loss��>

reg_losslw<


total_loss�@�>


accuracy_1q=
?>>��]       a[��	F,_��A�*O

prediction_loss���>

reg_losslw<


total_loss�}�>


accuracy_1�?�む]       a[��	�4,_��A�*O

prediction_loss   ?

reg_loss	lw<


total_loss��?


accuracy_1   ?�;�]       a[��	9c,_��A�*O

prediction_loss�G�>

reg_loss�kw<


total_loss�>


accuracy_1)\?B#�]       a[��	�,_��A�*O

prediction_loss�G�>

reg_loss�kw<


total_loss�>


accuracy_1)\?���]       a[��	$�,_��A�*O

prediction_loss�z?

reg_loss�kw<


total_loss�X?


accuracy_1=
�><���]       a[��	��,_��A�*O

prediction_loss�G�>

reg_loss�kw<


total_loss�>


accuracy_1)\?2��]       a[��	R(,_��A�*O

prediction_loss�Q�>

reg_loss�kw<


total_lossK�>


accuracy_1
�#?�RKF]       a[��	�P,_��A�*O

prediction_loss��>

reg_loss�kw<


total_loss}@�>


accuracy_1q=
?�1�]       a[��	i�,_��A�*O

prediction_loss�?

reg_loss�kw<


total_lossg�?


accuracy_1���>���]       a[��	��,_��A�*O

prediction_loss���>

reg_loss�kw<


total_loss�}�>


accuracy_1�?��U]       a[��	��,_��A�*O

prediction_loss=
�>

reg_loss�kw<


total_loss���>


accuracy_1�z?��]       a[��	�	,_��A�*O

prediction_loss�Q�>

reg_loss�kw<


total_lossI�>


accuracy_1
�#?!'o]       a[��	�(,_��A�*O

prediction_loss�?

reg_loss�kw<


total_lossf�?


accuracy_1���>�AXn]       a[��	jP,_��A�*O

prediction_loss�z?

reg_loss�kw<


total_loss�X?


accuracy_1=
�>3�cL]       a[��	�q,_��A�*O

prediction_loss���>

reg_loss�kw<


total_loss�}�>


accuracy_1�?���]       a[��	��,_��A�*O

prediction_loss)\?

reg_lossxkw<


total_loss�9?


accuracy_1�G�>|\��]       a[��	ͱ,_��A�*O

prediction_loss=
�>

reg_lossnkw<


total_loss���>


accuracy_1�z?vֵ�]       a[��	6�,_��A�*O

prediction_loss=
�>

reg_lossdkw<


total_loss���>


accuracy_1�z?��]       a[��	��,_��A�*O

prediction_loss��>

reg_lossYkw<


total_lossz@�>


accuracy_1q=
?��2e]       a[��	�,_��A�*O

prediction_loss�?

reg_lossPkw<


total_losse�?


accuracy_1���>�s�$]       a[��	!,_��A�*O

prediction_loss�z?

reg_lossEkw<


total_loss�X?


accuracy_1=
�>�P�]       a[��	B>,_��A�*O

prediction_loss��?

reg_loss9kw<


total_lossGw?


accuracy_1���>[��]       a[��	!Z,_��A�*O

prediction_loss   ?

reg_loss.kw<


total_loss��?


accuracy_1   ?Q��]       a[��	w,_��A�*O

prediction_loss��>

reg_loss$kw<


total_lossx@�>


accuracy_1q=
?��st]       a[��	��,_��A�*O

prediction_loss)\?

reg_losskw<


total_loss�9?


accuracy_1�G�>��4�]       a[��	ޭ,_��A�*O

prediction_loss)\?

reg_losskw<


total_loss�9?


accuracy_1�G�>�ދ]       a[��	��,_��A�*O

prediction_loss���>

reg_losskw<


total_loss�}�>


accuracy_1�?�5�N]       a[��	7�,_��A�*O

prediction_lossq=
?

reg_loss�jw<


total_loss?


accuracy_1��>�mh�]       a[��	� ,_��A�*O

prediction_loss   ?

reg_loss�jw<


total_loss��?


accuracy_1   ?����]       a[��	�,_��A�*O

prediction_loss
�#?

reg_loss�jw<


total_loss��'?


accuracy_1�Q�>!9�=]       a[��	�4,_��A�*O

prediction_loss\��>

reg_loss�jw<


total_loss�J�>


accuracy_1R�?���V]       a[��	<O,_��A�*O

prediction_lossq=
?

reg_loss�jw<


total_loss?


accuracy_1��>tbH(]       a[��	g,_��A�*O

prediction_loss�z?

reg_loss�jw<


total_loss�X?


accuracy_1=
�>��^3]       a[��	˃,_��A�*O

prediction_loss=
�>

reg_loss�jw<


total_loss���>


accuracy_1�z?=&�]       a[��	��,_��A�*O

prediction_loss�G�>

reg_loss�jw<


total_loss�>


accuracy_1)\?�/��]       a[��	�,_��A�*O

prediction_loss)\?

reg_loss�jw<


total_loss�9?


accuracy_1�G�>YX��]       a[��	�,_��A�*O

prediction_loss   ?

reg_loss�jw<


total_loss��?


accuracy_1   ?1�L[]       a[��	��,_��A�*O

prediction_loss���>

reg_loss�jw<


total_loss�}�>


accuracy_1�?��pb]       a[��	A,_��A�*O

prediction_loss�G�>

reg_loss�jw<


total_loss�>


accuracy_1)\?[��]       a[��	28,_��A�*O

prediction_loss
�#?

reg_loss�jw<


total_loss��'?


accuracy_1�Q�>A�@m]       a[��	�],_��A�*O

prediction_loss   ?

reg_lossvjw<


total_loss��?


accuracy_1   ?!���]       a[��	+�,_��A�*O

prediction_loss��>

reg_lossljw<


total_lossr@�>


accuracy_1q=
?k���]       a[��	ū,_��A�*O

prediction_loss�G�>

reg_lossajw<


total_loss�>


accuracy_1)\?ڳB]       a[��	��,_��A�*O

prediction_loss���>

reg_lossWjw<


total_loss�}�>


accuracy_1�?5	]       a[��	!�,_��A�*O

prediction_loss�G�>

reg_lossMjw<


total_loss �>


accuracy_1)\?�ROe]       a[��	i,_��A�*O

prediction_loss���>

reg_lossAjw<


total_loss��>


accuracy_1��?��#�]       a[��	�S,_��A�*O

prediction_loss=
�>

reg_loss8jw<


total_loss���>


accuracy_1�z?�bp�]       a[��	�v,_��A�*O

prediction_loss��?

reg_loss.jw<


total_lossCw?


accuracy_1���>�]z]       a[��	��,_��A�*O

prediction_lossq=
?

reg_loss%jw<


total_loss?


accuracy_1��>����]       a[��	�,_��A�*O

prediction_loss)\?

reg_lossjw<


total_loss�9?


accuracy_1�G�>��?K]       a[��	��,_��A�*O

prediction_loss��>

reg_lossjw<


total_lossp@�>


accuracy_1q=
?>��j]       a[��	�6,_��A�*O

prediction_lossq=
?

reg_lossjw<


total_loss?


accuracy_1��>"��]       a[��	t\,_��A�*O

prediction_lossR�?

reg_loss�iw<


total_loss��"?


accuracy_1\��>z�]       a[��	��,_��A�*O

prediction_loss   ?

reg_loss�iw<


total_loss��?


accuracy_1   ?U�W�]       a[��	\�,_��A�*O

prediction_loss�G�>

reg_loss�iw<


total_loss��>


accuracy_1)\?�'�]       a[��	o�,_��A�*O

prediction_loss�?

reg_loss�iw<


total_loss_�?


accuracy_1���>��EU]       a[��	�7,_��A�*O

prediction_loss\��>

reg_loss�iw<


total_loss�J�>


accuracy_1R�?ɲt)]       a[��	�Y,_��A�*O

prediction_loss��>

reg_loss�iw<


total_lossm@�>


accuracy_1q=
?!���]       a[��	��,_��A�*O

prediction_loss���>

reg_loss�iw<


total_loss��>


accuracy_1��?6�kv]       a[��	d�,_��A�*O

prediction_loss=
�>

reg_loss�iw<


total_loss���>


accuracy_1�z?hQX]       a[��	~�,_��A�*O

prediction_loss)\?

reg_loss�iw<


total_loss�9?


accuracy_1�G�>��T]       a[��	�$,_��A�*O

prediction_loss�?

reg_loss�iw<


total_loss_�?


accuracy_1���>0/��]       a[��	2Y,_��A�*O

prediction_loss��>

reg_loss�iw<


total_lossl@�>


accuracy_1q=
?i��]       a[��	�x,_��A�*O

prediction_loss�z?

reg_loss�iw<


total_loss�X?


accuracy_1=
�>ཞ�]       a[��	��,_��A�*O

prediction_loss��>

reg_loss�iw<


total_lossk@�>


accuracy_1q=
?+�-�]       a[��	-�,_��A�*O

prediction_loss�z?

reg_lossyiw<


total_loss�X?


accuracy_1=
�>d�6M]       a[��	1�,_��A�*O

prediction_loss)\?

reg_lossoiw<


total_loss�9?


accuracy_1�G�>���]       a[��	I�,_��A�*O

prediction_loss��>

reg_lossdiw<


total_lossj@�>


accuracy_1q=
?�!��]       a[��	,_��A�*O

prediction_loss���>

reg_lossYiw<


total_loss�}�>


accuracy_1�?M65]       a[��	�G,_��A�*O

prediction_loss��>

reg_lossOiw<


total_lossi@�>


accuracy_1q=
?ϫ7�]       a[��	&r,_��A�*O

prediction_loss�?

reg_lossGiw<


total_loss]�?


accuracy_1���>�8�]       a[��	֐,_��A�*O

prediction_loss{�>

reg_loss<iw<


total_loss�ϵ>


accuracy_1��(?{��]       a[��	�,_��A�*O

prediction_loss��>

reg_loss1iw<


total_lossi@�>


accuracy_1q=
?]��]       a[��	��,_��A�*O

prediction_loss   ?

reg_loss%iw<


total_loss��?


accuracy_1   ? 0��]       a[��	��,_��A�*O

prediction_loss�?

reg_lossiw<


total_loss\�?


accuracy_1���>��e]       a[��	�,_��A�*O

prediction_loss)\�>

reg_lossiw<


total_lossr�>


accuracy_1�Q8?^g�]       a[��	�',_��A�*O

prediction_loss
�#?

reg_lossiw<


total_loss��'?


accuracy_1�Q�>5Fe]       a[��	��,_��A�*O

prediction_loss�?

reg_loss�hw<


total_loss\�?


accuracy_1���>Y-]       a[��	��,_��A�*O

prediction_loss\��>

reg_loss�hw<


total_loss�J�>


accuracy_1R�?]�� ]       a[��	�,_��A�*O

prediction_lossq=
?

reg_loss�hw<


total_loss?


accuracy_1��>&v]       a[��	��,_��A�*O

prediction_loss   ?

reg_loss�hw<


total_loss��?


accuracy_1   ?���.]       a[��	>,_��A�*O

prediction_loss��>

reg_loss�hw<


total_losse@�>


accuracy_1q=
?͹�m]       a[��	�,_��A�*O

prediction_loss���>

reg_loss�hw<


total_loss�}�>


accuracy_1�?���]       a[��	�8,_��A�*O

prediction_loss
�#?

reg_loss�hw<


total_loss��'?


accuracy_1�Q�>�c^']       a[��	PQ,_��A�*O

prediction_loss��?

reg_loss�hw<


total_loss=w?


accuracy_1���>S���]       a[��	]l,_��A�*O

prediction_loss)\?

reg_loss�hw<


total_loss�9?


accuracy_1�G�>Z���]       a[��	ׅ,_��A�*O

prediction_loss��?

reg_loss�hw<


total_loss<w?


accuracy_1���>|8�M]       a[��	��,_��A�*O

prediction_loss   ?

reg_loss�hw<


total_loss��?


accuracy_1   ?q���]       a[��	��,_��A�*O

prediction_loss   ?

reg_losshw<


total_loss��?


accuracy_1   ?��]       a[��	'�,_��A�*O

prediction_loss���>

reg_lossuhw<


total_loss�}�>


accuracy_1�?zG-x]       a[��	U�,_��A�*O

prediction_loss���>

reg_lossjhw<


total_loss��>


accuracy_1��?�~��]       a[��	� ,_��A�*O

prediction_loss���>

reg_loss`hw<


total_loss�}�>


accuracy_1�?j>�]       a[��	�) ,_��A�*O

prediction_loss�z?

reg_lossUhw<


total_loss�X?


accuracy_1=
�>��!�]       a[��	B ,_��A�*O

prediction_loss���>

reg_lossKhw<


total_loss�}�>


accuracy_1�?Zt]       a[��	�_ ,_��A�*O

prediction_loss���>

reg_loss@hw<


total_loss�}�>


accuracy_1�?�i�[]       a[��	�� ,_��A�*O

prediction_loss�G�>

reg_loss6hw<


total_loss��>


accuracy_1)\?y��]       a[��	�� ,_��A�*O

prediction_loss�?

reg_loss-hw<


total_lossY�?


accuracy_1���>db'm]       a[��	� ,_��A�*O

prediction_loss   ?

reg_loss$hw<


total_loss��?


accuracy_1   ?;<,]       a[��	�!,_��A�*O

prediction_loss�G�>

reg_losshw<


total_loss��>


accuracy_1)\?�=�]       a[��	4!,_��A�*O

prediction_loss��?

reg_losshw<


total_loss:w?


accuracy_1���>��]       a[��	�~!,_��A�*O

prediction_lossq=
?

reg_losshw<


total_loss?


accuracy_1��>.��Q]       a[��	=�!,_��A�*O

prediction_lossq=
?

reg_loss�gw<


total_loss?


accuracy_1��>EWP]       a[��	k�!,_��A�*O

prediction_loss��?

reg_loss�gw<


total_loss:w?


accuracy_1���>��Ig]       a[��	,�!,_��A�*O

prediction_loss)\�>

reg_loss�gw<


total_lossh�>


accuracy_1�Q8?B<�F]       a[��	��!,_��A�*O

prediction_loss=
�>

reg_loss�gw<


total_loss|��>


accuracy_1�z?�چ]       a[��	�",_��A�*O

prediction_loss)\?

reg_loss�gw<


total_loss�9?


accuracy_1�G�>S҆�]       a[��	+",_��A�*O

prediction_loss   ?

reg_loss�gw<


total_loss��?


accuracy_1   ?�El�]       a[��	II",_��A�*O

prediction_loss)\?

reg_loss�gw<


total_loss�9?


accuracy_1�G�>2b�]       a[��	�i",_��A�*O

prediction_loss�?

reg_loss�gw<


total_lossW�?


accuracy_1���>1���]       a[��	��",_��A�*O

prediction_lossq=
?

reg_loss�gw<


total_loss?


accuracy_1��>Hx�]       a[��	��",_��A�*O

prediction_loss   ?

reg_loss�gw<


total_loss��?


accuracy_1   ?��k]       a[��	f�",_��A�*O

prediction_loss   ?

reg_loss�gw<


total_loss��?


accuracy_1   ?�sQ�]       a[��	�",_��A�*O

prediction_loss���>

reg_loss�gw<


total_loss�}�>


accuracy_1�?:$�S]       a[��	��",_��A�*O

prediction_loss   ?

reg_lossgw<


total_loss��?


accuracy_1   ?����]       a[��	�#,_��A�*O

prediction_lossq=
?

reg_losstgw<


total_loss?


accuracy_1��>ڱ��]       a[��	�/#,_��A�*O

prediction_loss=
�>

reg_losskgw<


total_lossx��>


accuracy_1�z?IG�]       a[��	MJ#,_��A�*O

prediction_loss��?

reg_loss_gw<


total_loss7w?


accuracy_1���>�oE�]       a[��	ge#,_��A�*O

prediction_loss��>

reg_lossVgw<


total_lossZ@�>


accuracy_1q=
?ַA�]       a[��	�#,_��A�*O

prediction_loss��>

reg_lossLgw<


total_lossY@�>


accuracy_1q=
?B��]       a[��	��#,_��A�*O

prediction_lossR�?

reg_loss@gw<


total_loss�"?


accuracy_1\��>��K{]       a[��	��#,_��A�*O

prediction_loss   ?

reg_loss7gw<


total_loss��?


accuracy_1   ?�ֻ�]       a[��	��#,_��A�*O

prediction_loss���>

reg_loss+gw<


total_loss�}�>


accuracy_1�?'Þ�]       a[��	\�#,_��A�*O

prediction_loss��>

reg_loss"gw<


total_lossX@�>


accuracy_1q=
?��}�]       a[��	�$,_��A�*O

prediction_loss   ?

reg_lossgw<


total_loss��?


accuracy_1   ?^/σ]       a[��	�$,_��A�*O

prediction_loss   ?

reg_lossgw<


total_loss��?


accuracy_1   ?�+i]       a[��	v6$,_��A�*O

prediction_loss��>

reg_lossgw<


total_lossW@�>


accuracy_1q=
?�x��]       a[��	�Q$,_��A�*O

prediction_loss���>

reg_loss�fw<


total_loss��>


accuracy_1��?�x�>]       a[��	ro$,_��A�*O

prediction_loss\��>

reg_loss�fw<


total_loss�J�>


accuracy_1R�?�i$�]       a[��	�$,_��A�*O

prediction_lossq=
?

reg_loss�fw<


total_loss?


accuracy_1��>�3_�]       a[��	/�$,_��A�*O

prediction_lossR�?

reg_loss�fw<


total_loss�"?


accuracy_1\��>��0]       a[��	��$,_��A�*O

prediction_loss��>

reg_loss�fw<


total_lossU@�>


accuracy_1q=
?Z#/]       a[��	��$,_��A�*O

prediction_loss�?

reg_loss�fw<


total_lossS�?


accuracy_1���>DwU]       a[��	��$,_��A�*O

prediction_loss�G�>

reg_loss�fw<


total_loss��>


accuracy_1)\?���]       a[��	%,_��A�*O

prediction_lossq=
?

reg_loss�fw<


total_loss?


accuracy_1��>�#]       a[��	 %,_��A�*O

prediction_loss���>

reg_loss�fw<


total_loss�}�>


accuracy_1�?°�#]       a[��	�6%,_��A�*O

prediction_loss���>

reg_loss�fw<


total_loss��>


accuracy_1��?�J0[]       a[��	/N%,_��A�*O

prediction_loss���>

reg_loss�fw<


total_loss��>


accuracy_1��?&{�]       a[��	wf%,_��A�*O

prediction_loss���>

reg_loss�fw<


total_loss�}�>


accuracy_1�?V�߉]       a[��	0�%,_��A�*O

prediction_loss)\?

reg_loss{fw<


total_loss�9?


accuracy_1�G�>(:��]       a[��	R�%,_��A�*O

prediction_loss��>

reg_losspfw<


total_lossR@�>


accuracy_1q=
?�s��]       a[��	p�%,_��A�*O

prediction_loss�G�>

reg_lossffw<


total_loss��>


accuracy_1)\?�9:�]       a[��	C�%,_��A�*O

prediction_loss=
�>

reg_loss\fw<


total_lossp��>


accuracy_1�z?ϸI�]       a[��	�%,_��A�*O

prediction_loss��>

reg_lossTfw<


total_lossR@�>


accuracy_1q=
?f��]       a[��	�&,_��A�*O

prediction_loss   ?

reg_lossHfw<


total_loss��?


accuracy_1   ?��:�]       a[��	�!&,_��A�*O

prediction_loss��>

reg_loss?fw<


total_lossQ@�>


accuracy_1q=
?��]       a[��	T:&,_��A�*O

prediction_loss   ?

reg_loss4fw<


total_loss��?


accuracy_1   ?3�$4]       a[��	�U&,_��A�*O

prediction_loss���>

reg_loss*fw<


total_loss�}�>


accuracy_1�?kS��]       a[��	�n&,_��A�*O

prediction_loss�?

reg_loss fw<


total_lossP�?


accuracy_1���>�Y�e]       a[��	�&,_��A�*O

prediction_lossq=
?

reg_lossfw<


total_loss	?


accuracy_1��>��]       a[��	��&,_��A�*O

prediction_loss)\?

reg_lossfw<


total_loss�9?


accuracy_1�G�>���]       a[��	��&,_��A�*O

prediction_lossq=
?

reg_lossfw<


total_loss	?


accuracy_1��>z��]       a[��	,�&,_��A�*O

prediction_lossq=
?

reg_loss�ew<


total_loss	?


accuracy_1��>>^�7]       a[��	�&,_��A�*O

prediction_loss�?

reg_loss�ew<


total_lossP�?


accuracy_1���>M�g]       a[��	',_��A�*O

prediction_loss�?

reg_loss�ew<


total_lossP�?


accuracy_1���>\�&�]       a[��	',_��A�*O

prediction_loss��?

reg_loss�ew<


total_loss1w?


accuracy_1���>���D]       a[��	7',_��A�*O

prediction_loss   ?

reg_loss�ew<


total_loss��?


accuracy_1   ?�p�]       a[��	�M',_��A�*O

prediction_loss�?

reg_loss�ew<


total_lossO�?


accuracy_1���>$��]       a[��	f',_��A�*O

prediction_lossR�?

reg_loss�ew<


total_loss�"?


accuracy_1\��>�#[]       a[��	,~',_��A�*O

prediction_loss���>

reg_loss�ew<


total_loss���>


accuracy_1��?-���]       a[��	��',_��A�*O

prediction_loss)\?

reg_loss�ew<


total_loss�9?


accuracy_1�G�>����]       a[��	��',_��A�*O

prediction_loss=
�>

reg_loss�ew<


total_lossj��>


accuracy_1�z?���V]       a[��	��',_��A�*O

prediction_loss��>

reg_loss�ew<


total_lossL@�>


accuracy_1q=
?ɡ)]       a[��	��',_��A�*O

prediction_loss   ?

reg_loss�ew<


total_loss��?


accuracy_1   ?��d ]       a[��	��',_��A�*O

prediction_loss�?

reg_loss{ew<


total_lossN�?


accuracy_1���>q]       a[��	R(,_��A�*O

prediction_loss���>

reg_lossqew<


total_loss�}�>


accuracy_1�?bqxy]       a[��	�'(,_��A�*O

prediction_loss��(?

reg_losshew<


total_lossY�,?


accuracy_1{�>���[]       a[��	cC(,_��A�*O

prediction_loss�?

reg_loss\ew<


total_lossM�?


accuracy_1���>��=]       a[��	�_(,_��A�*O

prediction_loss�G�>

reg_lossRew<


total_loss��>


accuracy_1)\? ��]       a[��	Sw(,_��A�*O

prediction_loss�?

reg_lossGew<


total_lossM�?


accuracy_1���>V@@�]       a[��	�(,_��A�*O

prediction_loss��?

reg_loss>ew<


total_loss/w?


accuracy_1���>�hE]       a[��		�(,_��A�*O

prediction_loss���>

reg_loss3ew<


total_loss�}�>


accuracy_1�?���}]       a[��	��(,_��A�*O

prediction_loss��>

reg_loss(ew<


total_lossH@�>


accuracy_1q=
?s��2]       a[��	+�(,_��A� *O

prediction_loss��>

reg_lossew<


total_lossH@�>


accuracy_1q=
?�4'�]       a[��	��(,_��A� *O

prediction_loss�z?

reg_lossew<


total_lossuX?


accuracy_1=
�>3�M]       a[��	�),_��A� *O

prediction_loss�z?

reg_loss
ew<


total_lossuX?


accuracy_1=
�>ƪ�]       a[��	7),_��A� *O

prediction_loss�z?

reg_loss�dw<


total_lossuX?


accuracy_1=
�>�fR�]       a[��	�1),_��A� *O

prediction_loss�G�>

reg_loss�dw<


total_loss��>


accuracy_1)\?�\D]       a[��	��),_��A� *O

prediction_loss��>

reg_loss�dw<


total_lossF@�>


accuracy_1q=
?6�E�]       a[��	U�),_��A� *O

prediction_loss   ?

reg_loss�dw<


total_loss��?


accuracy_1   ?��]       a[��	��),_��A� *O

prediction_loss���>

reg_loss�dw<


total_loss�}�>


accuracy_1�?ui�~]       a[��	��),_��A� *O

prediction_loss�G�>

reg_loss�dw<


total_loss��>


accuracy_1)\?ri,�]       a[��	!�),_��A� *O

prediction_lossq=
?

reg_loss�dw<


total_loss?


accuracy_1��>Q���]       a[��	� *,_��A� *O

prediction_loss�?

reg_loss�dw<


total_lossK�?


accuracy_1���>]8��]       a[��	�*,_��A� *O

prediction_loss��>

reg_loss�dw<


total_lossD@�>


accuracy_1q=
?���Y]       a[��	�4*,_��A� *O

prediction_loss�?

reg_loss�dw<


total_lossJ�?


accuracy_1���>�pBV]       a[��	�J*,_��A� *O

prediction_loss�G�>

reg_loss�dw<


total_loss��>


accuracy_1)\?M��]       a[��	�b*,_��A� *O

prediction_loss���>

reg_loss�dw<


total_loss��>


accuracy_1��?�Ұ]       a[��	z*,_��A� *O

prediction_loss��>

reg_losszdw<


total_lossC@�>


accuracy_1q=
?ћ�|]       a[��	%�*,_��A� *O

prediction_loss��>

reg_lossodw<


total_lossB@�>


accuracy_1q=
?�k�]       a[��	C�*,_��A� *O

prediction_loss)\?

reg_lossedw<


total_loss�9?


accuracy_1�G�>i��]       a[��	��*,_��A� *O

prediction_loss�z?

reg_lossZdw<


total_lossrX?


accuracy_1=
�>hm9�]       a[��	@�*,_��A� *O

prediction_loss�z?

reg_lossQdw<


total_lossrX?


accuracy_1=
�>F�m�]       a[��	 �*,_��A� *O

prediction_loss=
�>

reg_lossEdw<


total_loss_��>


accuracy_1�z?��e�]       a[��	B+,_��A� *O

prediction_loss��?

reg_loss<dw<


total_loss+w?


accuracy_1���>��%]       a[��	�+,_��A� *O

prediction_loss   ?

reg_loss/dw<


total_loss��?


accuracy_1   ?S�^�]       a[��	D3+,_��A� *O

prediction_loss��?

reg_loss'dw<


total_loss+w?


accuracy_1���>}J w]       a[��	jK+,_��A� *O

prediction_loss)\?

reg_lossdw<


total_loss�9?


accuracy_1�G�>���]       a[��	8g+,_��A� *O

prediction_loss�Q�>

reg_lossdw<


total_loss�>


accuracy_1
�#?sN/�]       a[��	��+,_��A� *O

prediction_loss   ?

reg_lossdw<


total_loss��?


accuracy_1   ?TQ�#]       a[��	x�+,_��A� *O

prediction_loss{.?

reg_loss�cw<


total_loss�1?


accuracy_1
ף>{"[�]       a[��	��+,_��A� *O

prediction_loss��>

reg_loss�cw<


total_loss?@�>


accuracy_1q=
?��Z�]       a[��	��+,_��A� *O

prediction_loss��>

reg_loss�cw<


total_loss>@�>


accuracy_1q=
?wQ��]       a[��	j,,_��A� *O

prediction_lossq=
?

reg_loss�cw<


total_loss ?


accuracy_1��>E΅]       a[��	�m,,_��A� *O

prediction_lossq=
?

reg_loss�cw<


total_loss ?


accuracy_1��>hD�	]       a[��	�,,_��A� *O

prediction_loss�z?

reg_loss�cw<


total_losspX?


accuracy_1=
�>���2]       a[��	��,,_��A� *O

prediction_loss��>

reg_loss�cw<


total_loss=@�>


accuracy_1q=
?ף}�]       a[��	��,,_��A� *O

prediction_loss=
�>

reg_loss�cw<


total_loss[��>


accuracy_1�z?�r�$]       a[��	��,,_��A� *O

prediction_lossR�?

reg_loss�cw<


total_loss�"?


accuracy_1\��>�z]       a[��	�-,_��A� *O

prediction_loss   ?

reg_loss�cw<


total_loss��?


accuracy_1   ?E�h]       a[��	-#-,_��A� *O

prediction_loss�?

reg_loss�cw<


total_lossF�?


accuracy_1���>��!�]       a[��	?-,_��A� *O

prediction_loss)\?

reg_loss�cw<


total_loss�9?


accuracy_1�G�>E���]       a[��	`Z-,_��A� *O

prediction_loss��>

reg_loss�cw<


total_loss;@�>


accuracy_1q=
?���]       a[��	z-,_��A� *O

prediction_loss\��>

reg_lossvcw<


total_lossxJ�>


accuracy_1R�?�#�N]       a[��	y�-,_��A� *O

prediction_loss��?

reg_losslcw<


total_loss(w?


accuracy_1���>xtL�]       a[��	�-,_��A� *O

prediction_loss���>

reg_lossbcw<


total_loss�}�>


accuracy_1�?x�$o]       a[��	��-,_��A� *O

prediction_loss   ?

reg_lossWcw<


total_loss��?


accuracy_1   ?��ħ]       a[��	��-,_��A� *O

prediction_loss�?

reg_lossMcw<


total_lossE�?


accuracy_1���>*�]       a[��	�.,_��A� *O

prediction_loss�z?

reg_lossAcw<


total_lossnX?


accuracy_1=
�>#��]       a[��	(.,_��A� *O

prediction_lossq=
?

reg_loss7cw<


total_loss�?


accuracy_1��>�5j�]       a[��	<J.,_��A� *O

prediction_loss�?

reg_loss,cw<


total_lossE�?


accuracy_1���>d�8l]       a[��	�d.,_��A� *O

prediction_loss��>

reg_loss$cw<


total_loss8@�>


accuracy_1q=
?�v]       a[��	(~.,_��A� *O

prediction_loss���>

reg_losscw<


total_loss��>


accuracy_1��?��C]       a[��	��.,_��A� *O

prediction_loss�?

reg_losscw<


total_lossD�?


accuracy_1���>D�e]       a[��	p�.,_��A� *O

prediction_loss�Q�>

reg_losscw<


total_loss�>


accuracy_1
�#?�kę]       a[��	-�.,_��A� *O

prediction_loss��>

reg_loss�bw<


total_loss7@�>


accuracy_1q=
?��]       a[��	�.,_��A� *O

prediction_loss���>

reg_loss�bw<


total_loss�}�>


accuracy_1�?%�e]       a[��	�/,_��A� *O

prediction_loss�?

reg_loss�bw<


total_lossD�?


accuracy_1���>�J�L]       a[��	�"/,_��A� *O

prediction_loss�?

reg_loss�bw<


total_lossC�?


accuracy_1���>p��]       a[��	�>/,_��A� *O

prediction_loss)\?

reg_loss�bw<


total_loss�9?


accuracy_1�G�>���]       a[��	GU/,_��A� *O

prediction_loss�?

reg_loss�bw<


total_lossC�?


accuracy_1���>�F]       a[��	Hl/,_��A� *O

prediction_loss�G�>

reg_loss�bw<


total_loss��>


accuracy_1)\?X5ق]       a[��	ׅ/,_��A� *O

prediction_loss   ?

reg_loss�bw<


total_loss��?


accuracy_1   ?�Wy*]       a[��	E�/,_��A� *O

prediction_loss�?

reg_loss�bw<


total_lossC�?


accuracy_1���>α��]       a[��	g�/,_��A� *O

prediction_loss=
�>

reg_loss�bw<


total_lossR��>


accuracy_1�z?��]       a[��	��/,_��A� *O

prediction_loss   ?

reg_loss�bw<


total_loss��?


accuracy_1   ?EBƦ]       a[��	��/,_��A� *O

prediction_loss���>

reg_loss�bw<


total_loss�}�>


accuracy_1�?x�G]       a[��	$0,_��A� *O

prediction_lossq=
?

reg_loss}bw<


total_loss�?


accuracy_1��>V}�B]       a[��	R,0,_��A� *O

prediction_loss��>

reg_losssbw<


total_loss3@�>


accuracy_1q=
?^|�m]       a[��	wH0,_��A� *O

prediction_loss)\?

reg_lossibw<


total_loss�9?


accuracy_1�G�>��9�]       a[��	�b0,_��A� *O

prediction_loss�?

reg_loss_bw<


total_lossA�?


accuracy_1���>���]       a[��	�{0,_��A� *O

prediction_loss{�>

reg_lossTbw<


total_loss�ϵ>


accuracy_1��(?��,�]       a[��	��0,_��A� *O

prediction_loss���>

reg_lossJbw<


total_loss߇�>


accuracy_1��?v5"]       a[��	*�0,_��A� *O

prediction_loss���>

reg_lossAbw<


total_loss߇�>


accuracy_1��?�j]       a[��	.�0,_��A� *O

prediction_loss�z?

reg_loss7bw<


total_lossjX?


accuracy_1=
�>���]       a[��	�0,_��A� *O

prediction_loss���>

reg_loss.bw<


total_loss�}�>


accuracy_1�?��Ǻ]       a[��	��0,_��A� *O

prediction_loss�?

reg_loss"bw<


total_lossA�?


accuracy_1���>����]       a[��	U1,_��A� *O

prediction_lossq=
?

reg_lossbw<


total_loss�?


accuracy_1��>F�C]       a[��	<21,_��A� *O

prediction_loss���>

reg_lossbw<


total_lossއ�>


accuracy_1��?vA�]       a[��	�J1,_��A� *O

prediction_loss   ?

reg_lossbw<


total_loss��?


accuracy_1   ?S�]       a[��	tb1,_��A� *O

prediction_loss�?

reg_loss�aw<


total_loss@�?


accuracy_1���>X`�]       a[��	yx1,_��A� *O

prediction_loss�G�>

reg_loss�aw<


total_loss��>


accuracy_1)\?%o�}]       a[��	q�1,_��A� *O

prediction_loss�z?

reg_loss�aw<


total_lossiX?


accuracy_1=
�>+ʏd]       a[��	¥1,_��A� *O

prediction_loss���>

reg_loss�aw<


total_loss܇�>


accuracy_1��?)�]       a[��	��1,_��A� *O

prediction_lossq=
?

reg_loss�aw<


total_loss�?


accuracy_1��>����]       a[��	��1,_��A� *O

prediction_loss�G�>

reg_loss�aw<


total_loss��>


accuracy_1)\?Z� ]       a[��	x�1,_��A� *O

prediction_loss)\?

reg_loss�aw<


total_loss�9?


accuracy_1�G�>xK|�]       a[��	B2,_��A� *O

prediction_lossq=
?

reg_loss�aw<


total_loss�?


accuracy_1��>h6K]       a[��	�2,_��A� *O

prediction_loss�G�>

reg_loss�aw<


total_loss��>


accuracy_1)\?��9�]       a[��	�42,_��A� *O

prediction_loss�?

reg_loss�aw<


total_loss>�?


accuracy_1���>Z-��]       a[��	�K2,_��A� *O

prediction_loss�G�>

reg_loss�aw<


total_loss��>


accuracy_1)\?�8��]       a[��	a2,_��A� *O

prediction_loss   ?

reg_loss�aw<


total_loss��?


accuracy_1   ?�v�/]       a[��	-y2,_��A� *O

prediction_loss�z?

reg_loss|aw<


total_lossgX?


accuracy_1=
�>���]       a[��	C�2,_��A� *O

prediction_lossq=
?

reg_losssaw<


total_loss�?


accuracy_1��>�p�]       a[��	ҧ2,_��A� *O

prediction_loss�?

reg_lossiaw<


total_loss>�?


accuracy_1���>��ڎ]       a[��	��2,_��A� *O

prediction_loss=
�>

reg_loss]aw<


total_lossH��>


accuracy_1�z?���4]       a[��	Z�2,_��A� *O

prediction_loss��>

reg_lossTaw<


total_loss*@�>


accuracy_1q=
?z�c]       a[��	��2,_��A� *O

prediction_loss)\?

reg_lossIaw<


total_loss�9?


accuracy_1�G�>��u�]       a[��	>3,_��A� *O

prediction_loss�?

reg_lossAaw<


total_loss=�?


accuracy_1���>)��]       a[��	�3,_��A� *O

prediction_loss��>

reg_loss5aw<


total_loss)@�>


accuracy_1q=
?�j�]       a[��	'23,_��A� *O

prediction_loss��?

reg_loss+aw<


total_lossw?


accuracy_1���>�_>0]       a[��	AJ3,_��A� *O

prediction_loss=
�>

reg_loss"aw<


total_lossF��>


accuracy_1�z?�M�]       a[��	�^3,_��A� *O

prediction_loss�Q�>

reg_lossaw<


total_loss��>


accuracy_1
�#?����]       a[��	Fx3,_��A� *O

prediction_loss   ?

reg_lossaw<


total_loss��?


accuracy_1   ?lx�]       a[��	��3,_��A� *O

prediction_loss�G�>

reg_lossaw<


total_loss��>


accuracy_1)\?,�*�]       a[��	�3,_��A� *O

prediction_loss�?

reg_loss�`w<


total_loss<�?


accuracy_1���>�2"�]       a[��	z�3,_��A� *O

prediction_loss�?

reg_loss�`w<


total_loss<�?


accuracy_1���>`N�]       a[��	3�3,_��A� *O

prediction_loss=
�>

reg_loss�`w<


total_lossD��>


accuracy_1�z?��M ]       a[��	-4,_��A� *O

prediction_loss   ?

reg_loss�`w<


total_loss��?


accuracy_1   ?(���]       a[��	�E4,_��A� *O

prediction_loss   ?

reg_loss�`w<


total_loss��?


accuracy_1   ?��o]       a[��	]4,_��A� *O

prediction_loss)\?

reg_loss�`w<


total_loss�9?


accuracy_1�G�>����]       a[��	Tt4,_��A� *O

prediction_loss{�>

reg_loss�`w<


total_loss�ϵ>


accuracy_1��(?Ր�J]       a[��	��4,_��A� *O

prediction_loss=
�>

reg_loss�`w<


total_lossB��>


accuracy_1�z?���]       a[��	��4,_��A� *O

prediction_lossq=
?

reg_loss�`w<


total_loss�?


accuracy_1��>G��]       a[��	4�4,_��A� *O

prediction_loss\��>

reg_loss�`w<


total_lossaJ�>


accuracy_1R�?��?<]       a[��	��4,_��A� *O

prediction_loss��?

reg_loss�`w<


total_lossw?


accuracy_1���>2�>]       a[��	2�4,_��A� *O

prediction_loss���>

reg_loss}`w<


total_loss�}�>


accuracy_1�?3�]       a[��	5,_��A� *O

prediction_loss=
�>

reg_lossp`w<


total_loss@��>


accuracy_1�z?��5?]       a[��	�5,_��A� *O

prediction_loss   ?

reg_lossg`w<


total_loss��?


accuracy_1   ?���]       a[��	�25,_��A� *O

prediction_loss=
�>

reg_lossZ`w<


total_loss@��>


accuracy_1�z?�'�2]       a[��	IJ5,_��A� *O

prediction_lossq=
?

reg_lossQ`w<


total_loss�?


accuracy_1��>���]       a[��	4e5,_��A� *O

prediction_loss�z?

reg_lossE`w<


total_lossbX?


accuracy_1=
�>R�1�]       a[��	Fz5,_��A� *O

prediction_loss)\?

reg_loss<`w<


total_loss�9?


accuracy_1�G�>�*]       a[��	�5,_��A� *O

prediction_loss�?

reg_loss2`w<


total_loss9�?


accuracy_1���>�5�L]       a[��	�5,_��A� *O

prediction_loss��>

reg_loss&`w<


total_loss @�>


accuracy_1q=
?�4%]       a[��	��5,_��A� *O

prediction_loss���>

reg_loss`w<


total_loss�}�>


accuracy_1�?:T`]       a[��	��5,_��A� *O

prediction_loss��>

reg_loss`w<


total_loss @�>


accuracy_1q=
?�ι]       a[��	g�5,_��A� *O

prediction_loss\��>

reg_loss`w<


total_loss\J�>


accuracy_1R�?FY��]       a[��	t6,_��A� *O

prediction_loss)\?

reg_loss�_w<


total_loss�9?


accuracy_1�G�>x�[]       a[��	]56,_��A� *O

prediction_loss���>

reg_loss�_w<


total_loss͇�>


accuracy_1��?��5]       a[��	$_6,_��A� *O

prediction_lossq=
?

reg_loss�_w<


total_loss�?


accuracy_1��>�@�]       a[��	σ6,_��A�!*O

prediction_loss�?

reg_loss�_w<


total_loss7�?


accuracy_1���>~�4]       a[��	�6,_��A�!*O

prediction_lossq=
?

reg_loss�_w<


total_loss�?


accuracy_1��>�R��]       a[��	<�6,_��A�!*O

prediction_loss=
�>

reg_loss�_w<


total_loss;��>


accuracy_1�z?f@M]       a[��	@�6,_��A�!*O

prediction_loss�?

reg_loss�_w<


total_loss7�?


accuracy_1���>H��]       a[��	%7,_��A�!*O

prediction_loss���>

reg_loss�_w<


total_loss�}�>


accuracy_1�?�Н]       a[��	J7,_��A�!*O

prediction_loss��>

reg_loss�_w<


total_loss@�>


accuracy_1q=
?YO.]       a[��	am7,_��A�!*O

prediction_loss=
�>

reg_loss�_w<


total_loss:��>


accuracy_1�z?ޙ7�]       a[��	��7,_��A�!*O

prediction_loss\�B?

reg_loss�_w<


total_loss�lF?


accuracy_1��u>�Q�]       a[��	ձ7,_��A�!*O

prediction_loss�G�>

reg_loss�_w<


total_loss��>


accuracy_1)\?0�]       a[��	.�7,_��A�!*O

prediction_loss���>

reg_loss�_w<


total_lossɇ�>


accuracy_1��?�}9]       a[��	>�7,_��A�!*O

prediction_lossq=
?

reg_lossv_w<


total_loss�?


accuracy_1��>���\]       a[��	18,_��A�!*O

prediction_loss)\?

reg_lossk_w<


total_loss�9?


accuracy_1�G�>��(]       a[��	�&8,_��A�!*O

prediction_lossq=
?

reg_lossb_w<


total_loss�?


accuracy_1��>��3�]       a[��	�C8,_��A�!*O

prediction_loss
ף>

reg_lossX_w<


total_loss��>


accuracy_1{.?��m�]       a[��	a8,_��A�!*O

prediction_loss���>

reg_lossO_w<


total_loss�}�>


accuracy_1�?�Z��]       a[��	_~8,_��A�!*O

prediction_loss
�#?

reg_lossB_w<


total_loss��'?


accuracy_1�Q�>�?�]       a[��	��8,_��A�!*O

prediction_loss)\?

reg_loss9_w<


total_loss�9?


accuracy_1�G�>�E�]       a[��	_�8,_��A�!*O

prediction_loss
�#?

reg_loss._w<


total_loss��'?


accuracy_1�Q�>��^]       a[��	��8,_��A�!*O

prediction_loss��?

reg_loss$_w<


total_lossw?


accuracy_1���>�Q~\]       a[��	��8,_��A�!*O

prediction_lossR�?

reg_loss_w<


total_lossΕ"?


accuracy_1\��>��xj]       a[��	K9,_��A�!*O

prediction_loss��?

reg_loss_w<


total_lossw?


accuracy_1���>�U]       a[��	*9,_��A�!*O

prediction_loss=
�>

reg_loss_w<


total_loss5��>


accuracy_1�z?͉'�]       a[��	S<9,_��A�!*O

prediction_loss�z?

reg_loss�^w<


total_loss]X?


accuracy_1=
�>]�;]       a[��	*X9,_��A�!*O

prediction_loss�?

reg_loss�^w<


total_loss4�?


accuracy_1���>5�$]       a[��	�w9,_��A�!*O

prediction_loss\��>

reg_loss�^w<


total_lossSJ�>


accuracy_1R�?���A]       a[��	)�9,_��A�!*O

prediction_loss�?

reg_loss�^w<


total_loss3�?


accuracy_1���>��]       a[��	u�9,_��A�!*O

prediction_loss���>

reg_loss�^w<


total_loss�}�>


accuracy_1�?���<]       a[��	��9,_��A�!*O

prediction_loss\��>

reg_loss�^w<


total_lossRJ�>


accuracy_1R�?wH�]       a[��	��9,_��A�!*O

prediction_loss�G�>

reg_loss�^w<


total_loss��>


accuracy_1)\?�m.�]       a[��	��9,_��A�!*O

prediction_loss   ?

reg_loss�^w<


total_loss{�?


accuracy_1   ?��m�]       a[��	X:,_��A�!*O

prediction_loss��?

reg_loss�^w<


total_lossw?


accuracy_1���>��Ɨ]       a[��	X7:,_��A�!*O

prediction_loss=
�>

reg_loss�^w<


total_loss2��>


accuracy_1�z?���s]       a[��	S:,_��A�!*O

prediction_loss�z?

reg_loss�^w<


total_loss[X?


accuracy_1=
�>�pR]       a[��	�m:,_��A�!*O

prediction_loss�G�>

reg_loss�^w<


total_loss��>


accuracy_1)\?WR�]       a[��	��:,_��A�!*O

prediction_loss���>

reg_loss~^w<


total_loss�}�>


accuracy_1�?j�b�]       a[��	��:,_��A�!*O

prediction_loss)\?

reg_losst^w<


total_loss�9?


accuracy_1�G�>��C]       a[��	��:,_��A�!*O

prediction_loss   ?

reg_lossj^w<


total_lossz�?


accuracy_1   ?׍N]       a[��	�:,_��A�!*O

prediction_loss�?

reg_loss_^w<


total_loss1�?


accuracy_1���>m.xe]       a[��	��:,_��A�!*O

prediction_loss���>

reg_lossT^w<


total_loss�}�>


accuracy_1�?���@]       a[��	;,_��A�!*O

prediction_loss�Q�>

reg_lossH^w<


total_loss��>


accuracy_1
�#?c�t]       a[��	�#;,_��A�!*O

prediction_loss   ?

reg_loss?^w<


total_lossy�?


accuracy_1   ?[X9�]       a[��	�<;,_��A�!*O

prediction_loss��(?

reg_loss6^w<


total_loss<�,?


accuracy_1{�>��\�]       a[��	�];,_��A�!*O

prediction_loss�?

reg_loss+^w<


total_loss1�?


accuracy_1���>\T�e]       a[��	�w;,_��A�!*O

prediction_loss��(?

reg_loss"^w<


total_loss<�,?


accuracy_1{�>���]       a[��	��;,_��A�!*O

prediction_loss   ?

reg_loss^w<


total_lossx�?


accuracy_1   ?�|�>]       a[��	��;,_��A�!*O

prediction_loss)\?

reg_loss^w<


total_loss�9?


accuracy_1�G�>��l]       a[��	��;,_��A�!*O

prediction_loss��(?

reg_loss^w<


total_loss;�,?


accuracy_1{�>���]       a[��	��;,_��A�!*O

prediction_loss�G�>

reg_loss�]w<


total_loss��>


accuracy_1)\?�� A]       a[��	U<,_��A�!*O

prediction_loss=
�>

reg_loss�]w<


total_loss,��>


accuracy_1�z?���]       a[��	�<<,_��A�!*O

prediction_loss��>

reg_loss�]w<


total_loss@�>


accuracy_1q=
?nk�]       a[��	`Y<,_��A�!*O

prediction_loss�?

reg_loss�]w<


total_loss/�?


accuracy_1���>�N��]       a[��	-�<,_��A�!*O

prediction_loss=
�>

reg_loss�]w<


total_loss+��>


accuracy_1�z?���]       a[��	��<,_��A�!*O

prediction_loss���>

reg_loss�]w<


total_loss���>


accuracy_1��?�3�]       a[��	L6=,_��A�!*O

prediction_loss���>

reg_loss�]w<


total_loss�T�>


accuracy_1333?�G ]       a[��	^=,_��A�!*O

prediction_loss��>

reg_loss�]w<


total_loss@�>


accuracy_1q=
?�O6b]       a[��	n�=,_��A�!*O

prediction_loss\��>

reg_loss�]w<


total_lossIJ�>


accuracy_1R�?���]       a[��	�=,_��A�!*O

prediction_loss��>

reg_loss�]w<


total_loss@�>


accuracy_1q=
?�W��]       a[��	]�=,_��A�!*O

prediction_lossq=
?

reg_loss�]w<


total_loss�?


accuracy_1��>dT&]       a[��	�>,_��A�!*O

prediction_loss�Q�>

reg_loss�]w<


total_loss��>


accuracy_1
�#?�s!]       a[��	P4>,_��A�!*O

prediction_loss��(?

reg_loss}]w<


total_loss9�,?


accuracy_1{�>|�X�]       a[��	�T>,_��A�!*O

prediction_loss=
�>

reg_losss]w<


total_loss)��>


accuracy_1�z?���[]       a[��	�}>,_��A�!*O

prediction_loss��>

reg_lossh]w<


total_loss
@�>


accuracy_1q=
?֓�]       a[��	.�>,_��A�!*O

prediction_loss���>

reg_loss]]w<


total_lossz}�>


accuracy_1�?c\]       a[��	$�>,_��A�!*O

prediction_loss���>

reg_lossU]w<


total_lossz}�>


accuracy_1�?y�E)]       a[��	W?,_��A�!*O

prediction_loss�?

reg_lossK]w<


total_loss-�?


accuracy_1���>�ǁR]       a[��	�;?,_��A�!*O

prediction_loss
�#?

reg_loss@]w<


total_loss�'?


accuracy_1�Q�>0Gk�]       a[��	9_?,_��A�!*O

prediction_loss�G�>

reg_loss4]w<


total_loss��>


accuracy_1)\?]^u�]       a[��	D�?,_��A�!*O

prediction_loss�?

reg_loss,]w<


total_loss-�?


accuracy_1���>[� ]       a[��	o�?,_��A�!*O

prediction_loss�?

reg_loss"]w<


total_loss-�?


accuracy_1���>�E�=]       a[��	7�?,_��A�!*O

prediction_loss���>

reg_loss]w<


total_lossx}�>


accuracy_1�?��,�]       a[��	�
@,_��A�!*O

prediction_loss   ?

reg_loss]w<


total_losst�?


accuracy_1   ?>3ţ]       a[��	<2@,_��A�!*O

prediction_lossq=
?

reg_loss]w<


total_loss�?


accuracy_1��>S���]       a[��	~V@,_��A�!*O

prediction_loss�?

reg_loss�\w<


total_loss,�?


accuracy_1���>��:]       a[��	V~@,_��A�!*O

prediction_loss��>

reg_loss�\w<


total_loss@�>


accuracy_1q=
?�E��]       a[��	?�@,_��A�!*O

prediction_loss   ?

reg_loss�\w<


total_losst�?


accuracy_1   ?x�#n]       a[��	$A,_��A�!*O

prediction_lossq=
?

reg_loss�\w<


total_loss�?


accuracy_1��>B"]       a[��	�KA,_��A�!*O

prediction_loss�?

reg_loss�\w<


total_loss+�?


accuracy_1���>��}]       a[��	��A,_��A�!*O

prediction_lossq=
?

reg_loss�\w<


total_loss�?


accuracy_1��>ex�]       a[��	K B,_��A�!*O

prediction_loss�z?

reg_loss�\w<


total_lossTX?


accuracy_1=
�>��&�]       a[��	�?B,_��A�!*O

prediction_loss=
�>

reg_loss�\w<


total_loss"��>


accuracy_1�z?��u�]       a[��	1}B,_��A�!*O

prediction_loss)\?

reg_loss�\w<


total_loss�9?


accuracy_1�G�>��8]       a[��	 �B,_��A�!*O

prediction_loss�?

reg_loss�\w<


total_loss*�?


accuracy_1���>D�{]       a[��	��B,_��A�!*O

prediction_lossq=
?

reg_loss�\w<


total_loss�?


accuracy_1��>ܨ�]       a[��	�C,_��A�!*O

prediction_lossq=
?

reg_loss\w<


total_loss�?


accuracy_1��>ߐ�j]       a[��	�`C,_��A�!*O

prediction_loss�z?

reg_lossr\w<


total_lossSX?


accuracy_1=
�>��r4]       a[��	+�C,_��A�!*O

prediction_loss=
�>

reg_lossi\w<


total_loss ��>


accuracy_1�z?L� ]       a[��	��C,_��A�!*O

prediction_lossq=
?

reg_loss^\w<


total_loss�?


accuracy_1��>4�=�]       a[��	�D,_��A�!*O

prediction_lossq=
?

reg_lossT\w<


total_loss�?


accuracy_1��>@[2]       a[��	=E,_��A�!*O

prediction_loss�?

reg_lossJ\w<


total_loss)�?


accuracy_1���>�Q�]       a[��	ԙE,_��A�!*O

prediction_loss���>

reg_loss?\w<


total_lossq}�>


accuracy_1�?���]       a[��	�E,_��A�!*O

prediction_loss�?

reg_loss5\w<


total_loss)�?


accuracy_1���>�* B]       a[��	��E,_��A�!*O

prediction_loss�z?

reg_loss,\w<


total_lossRX?


accuracy_1=
�>��G!]       a[��	�F,_��A�!*O

prediction_loss���>

reg_loss \w<


total_lossp}�>


accuracy_1�?�(&]       a[��	0.F,_��A�!*O

prediction_lossq=
?

reg_loss\w<


total_loss�?


accuracy_1��>&�B<]       a[��	jKF,_��A�!*O

prediction_loss=
�>

reg_loss\w<


total_loss��>


accuracy_1�z?�y�]       a[��	�qF,_��A�!*O

prediction_loss=
�>

reg_loss\w<


total_loss��>


accuracy_1�z?�Yt7]       a[��	;�F,_��A�!*O

prediction_loss   ?

reg_loss�[w<


total_lossp�?


accuracy_1   ?oxn]       a[��	��F,_��A�!*O

prediction_lossR�?

reg_loss�[w<


total_loss"?


accuracy_1\��>��V�]       a[��	��F,_��A�!*O

prediction_loss��>

reg_loss�[w<


total_loss�?�>


accuracy_1q=
?�� v]       a[��	GG,_��A�!*O

prediction_loss���>

reg_loss�[w<


total_loss���>


accuracy_1��?�!��]       a[��	�<G,_��A�!*O

prediction_loss�G�>

reg_loss�[w<


total_loss��>


accuracy_1)\?��,�]       a[��	 ^G,_��A�!*O

prediction_loss�G�>

reg_loss�[w<


total_loss��>


accuracy_1)\?�lt�]       a[��	��G,_��A�!*O

prediction_lossq=
?

reg_loss�[w<


total_loss�?


accuracy_1��>-g�I]       a[��	��G,_��A�!*O

prediction_loss���>

reg_loss�[w<


total_lossl}�>


accuracy_1�?�,_�]       a[��	��G,_��A�!*O

prediction_loss{�>

reg_loss�[w<


total_lossXϵ>


accuracy_1��(?�"Ϙ]       a[��	IH,_��A�!*O

prediction_loss���>

reg_loss�[w<


total_lossl}�>


accuracy_1�?�_]       a[��	�3H,_��A�!*O

prediction_loss�?

reg_loss�[w<


total_loss&�?


accuracy_1���>�y�]       a[��	$aH,_��A�!*O

prediction_loss   ?

reg_loss�[w<


total_lossn�?


accuracy_1   ?E�w]       a[��	̖H,_��A�!*O

prediction_loss���>

reg_loss{[w<


total_lossk}�>


accuracy_1�?�9�]       a[��	 I,_��A�!*O

prediction_loss�z?

reg_lossq[w<


total_lossOX?


accuracy_1=
�>�O8�]       a[��	f0I,_��A�!*O

prediction_loss�z?

reg_lossh[w<


total_lossOX?


accuracy_1=
�>�T!]       a[��	vTI,_��A�!*O

prediction_lossq=
?

reg_loss^[w<


total_loss�?


accuracy_1��>�RZ�]       a[��	Z�I,_��A�!*O

prediction_loss)\?

reg_lossS[w<


total_loss�9?


accuracy_1�G�>���]       a[��	>�I,_��A�!*O

prediction_loss��>

reg_lossH[w<


total_loss�?�>


accuracy_1q=
?2^A�]       a[��	N�I,_��A�!*O

prediction_loss�G�>

reg_loss@[w<


total_loss��>


accuracy_1)\?I�1]       a[��	��I,_��A�!*O

prediction_lossq=
?

reg_loss4[w<


total_loss�?


accuracy_1��>Ke��]       a[��	J,_��A�!*O

prediction_loss   ?

reg_loss*[w<


total_lossm�?


accuracy_1   ?f#�;]       a[��	�)J,_��A�!*O

prediction_loss)\?

reg_loss [w<


total_loss�9?


accuracy_1�G�>z�d�]       a[��	�KJ,_��A�!*O

prediction_loss��?

reg_loss[w<


total_lossw?


accuracy_1���>���]       a[��	�gJ,_��A�!*O

prediction_loss�z?

reg_loss[w<


total_lossMX?


accuracy_1=
�>���]       a[��	�J,_��A�!*O

prediction_loss�G�>

reg_loss[w<


total_loss��>


accuracy_1)\?X6H�]       a[��	��J,_��A�!*O

prediction_lossq=
?

reg_loss�Zw<


total_loss�?


accuracy_1��>�GK�]       a[��	q�J,_��A�!*O

prediction_loss��>

reg_loss�Zw<


total_loss�?�>


accuracy_1q=
?���H]       a[��	��J,_��A�!*O

prediction_loss���>

reg_loss�Zw<


total_lossf}�>


accuracy_1�?lJ�-]       a[��	�K,_��A�!*O

prediction_lossq=
?

reg_loss�Zw<


total_loss�?


accuracy_1��>6X�C]       a[��	�#K,_��A�!*O

prediction_loss   ?

reg_loss�Zw<


total_lossk�?


accuracy_1   ?�\��]       a[��	�EK,_��A�!*O

prediction_loss)\?

reg_loss�Zw<


total_loss�9?


accuracy_1�G�>��w]       a[��	�dK,_��A�!*O

prediction_loss)\?

reg_loss�Zw<


total_loss�9?


accuracy_1�G�>
���]       a[��	A~K,_��A�"*O

prediction_loss�z?

reg_loss�Zw<


total_lossLX?


accuracy_1=
�>���]       a[��	�K,_��A�"*O

prediction_loss   ?

reg_loss�Zw<


total_lossk�?


accuracy_1   ?�z�w]       a[��	��K,_��A�"*O

prediction_loss�G�>

reg_loss�Zw<


total_loss��>


accuracy_1)\?X�j�]       a[��	|�K,_��A�"*O

prediction_loss�?

reg_loss�Zw<


total_loss"�?


accuracy_1���>��#�]       a[��	�K,_��A�"*O

prediction_loss)\?

reg_loss�Zw<


total_loss�9?


accuracy_1�G�>�V_�]       a[��	mL,_��A�"*O

prediction_loss)\?

reg_loss{Zw<


total_loss�9?


accuracy_1�G�>�U�{]       a[��	�7L,_��A�"*O

prediction_loss�?

reg_lossrZw<


total_loss"�?


accuracy_1���>�A�]       a[��	SL,_��A�"*O

prediction_lossR�?

reg_lossgZw<


total_loss��"?


accuracy_1\��>N�g�]       a[��	erL,_��A�"*O

prediction_loss333?

reg_loss\Zw<


total_loss�7?


accuracy_1���>�f l]       a[��	h�L,_��A�"*O

prediction_loss��?

reg_lossTZw<


total_lossw?


accuracy_1���>�ޣ�]       a[��	��L,_��A�"*O

prediction_loss)\?

reg_lossIZw<


total_loss�9?


accuracy_1�G�>Aۓ]       a[��	z�L,_��A�"*O

prediction_loss�z?

reg_loss@Zw<


total_lossJX?


accuracy_1=
�>�M�z]       a[��	��L,_��A�"*O

prediction_loss   ?

reg_loss3Zw<


total_lossi�?


accuracy_1   ?[�-]       a[��	�M,_��A�"*O

prediction_loss   ?

reg_loss*Zw<


total_lossi�?


accuracy_1   ?����]       a[��	]4M,_��A�"*O

prediction_loss���>

reg_loss!Zw<


total_loss`}�>


accuracy_1�?�=��]       a[��	QM,_��A�"*O

prediction_loss�?

reg_lossZw<


total_loss �?


accuracy_1���>N�]       a[��	�nM,_��A�"*O

prediction_loss�?

reg_lossZw<


total_loss �?


accuracy_1���>{Mӌ]       a[��	�M,_��A�"*O

prediction_lossR�?

reg_lossZw<


total_loss��"?


accuracy_1\��>w�']       a[��	]�M,_��A�"*O

prediction_loss�?

reg_loss�Yw<


total_loss �?


accuracy_1���>ֺX�]       a[��	��M,_��A�"*O

prediction_lossq=
?

reg_loss�Yw<


total_loss�?


accuracy_1��>eR�]       a[��		�M,_��A�"*O

prediction_loss��>

reg_loss�Yw<


total_loss�?�>


accuracy_1q=
?:��]       a[��	��M,_��A�"*O

prediction_loss���>

reg_loss�Yw<


total_loss^}�>


accuracy_1�?��k�]       a[��	N,_��A�"*O

prediction_loss)\?

reg_loss�Yw<


total_loss�9?


accuracy_1�G�>cb]       a[��	�2N,_��A�"*O

prediction_loss�?

reg_loss�Yw<


total_loss�?


accuracy_1���>�D=�]       a[��	nLN,_��A�"*O

prediction_lossR�?

reg_loss�Yw<


total_loss��"?


accuracy_1\��>HC)�]       a[��	oeN,_��A�"*O

prediction_loss   ?

reg_loss�Yw<


total_lossg�?


accuracy_1   ?�@	�]       a[��	 ~N,_��A�"*O

prediction_loss)\?

reg_loss�Yw<


total_loss�9?


accuracy_1�G�>	��Y]       a[��	��N,_��A�"*O

prediction_loss��?

reg_loss�Yw<


total_loss w?


accuracy_1���>��]       a[��	I�N,_��A�"*O

prediction_loss   ?

reg_loss�Yw<


total_lossf�?


accuracy_1   ?��]       a[��	5�N,_��A�"*O

prediction_lossq=
?

reg_loss�Yw<


total_loss�?


accuracy_1��>#�2�]       a[��	]�N,_��A�"*O

prediction_loss   ?

reg_losswYw<


total_lossf�?


accuracy_1   ?8)yD]       a[��	�)O,_��A�"*O

prediction_loss�G�>

reg_lossmYw<


total_lossy�>


accuracy_1)\?)���]       a[��	rMO,_��A�"*O

prediction_loss�z?

reg_lossbYw<


total_lossGX?


accuracy_1=
�>��~�]       a[��	vmO,_��A�"*O

prediction_lossq=
?

reg_lossXYw<


total_loss�?


accuracy_1��>_��]       a[��	��O,_��A�"*O

prediction_loss=
�>

reg_lossMYw<


total_loss��>


accuracy_1�z?���&]       a[��	a�O,_��A�"*O

prediction_loss)\?

reg_lossDYw<


total_loss�9?


accuracy_1�G�>u��]       a[��	�O,_��A�"*O

prediction_loss)\?

reg_loss:Yw<


total_loss�9?


accuracy_1�G�>���]       a[��	��O,_��A�"*O

prediction_lossq=
?

reg_loss/Yw<


total_loss�?


accuracy_1��>�Y%]       a[��	2P,_��A�"*O

prediction_lossq=
?

reg_loss%Yw<


total_loss�?


accuracy_1��>��a]       a[��	�]P,_��A�"*O

prediction_loss   ?

reg_lossYw<


total_lossd�?


accuracy_1   ?~�)t]       a[��	֎P,_��A�"*O

prediction_loss�G�>

reg_lossYw<


total_lossv�>


accuracy_1)\?a'�]       a[��	��P,_��A�"*O

prediction_loss�?

reg_lossYw<


total_loss�?


accuracy_1���>7��]       a[��	��P,_��A�"*O

prediction_loss�z?

reg_loss�Xw<


total_lossEX?


accuracy_1=
�>��6�]       a[��	�P,_��A�"*O

prediction_loss)\?

reg_loss�Xw<


total_loss�9?


accuracy_1�G�>t�]       a[��	�Q,_��A�"*O

prediction_loss�z?

reg_loss�Xw<


total_lossEX?


accuracy_1=
�>�\p,]       a[��	U4Q,_��A�"*O

prediction_loss=
�>

reg_loss�Xw<


total_loss��>


accuracy_1�z?<�d{]       a[��	QQ,_��A�"*O

prediction_loss=
�>

reg_loss�Xw<


total_loss��>


accuracy_1�z?`�G�]       a[��	yxQ,_��A�"*O

prediction_loss   ?

reg_loss�Xw<


total_lossc�?


accuracy_1   ?����]       a[��	͔Q,_��A�"*O

prediction_loss)\?

reg_loss�Xw<


total_loss�9?


accuracy_1�G�>��@]       a[��	��Q,_��A�"*O

prediction_loss��>

reg_loss�Xw<


total_loss�?�>


accuracy_1q=
?����]       a[��	|R,_��A�"*O

prediction_loss\��>

reg_loss�Xw<


total_loss!J�>


accuracy_1R�?]Ug3]       a[��	�'R,_��A�"*O

prediction_loss�G�>

reg_loss�Xw<


total_losss�>


accuracy_1)\?��,�]       a[��	|FR,_��A�"*O

prediction_lossq=
?

reg_loss�Xw<


total_loss�?


accuracy_1��>�i��]       a[��	=`R,_��A�"*O

prediction_loss=
�>

reg_loss�Xw<


total_loss��>


accuracy_1�z?͜>�]       a[��	�|R,_��A�"*O

prediction_loss�?

reg_lossvXw<


total_loss�?


accuracy_1���>9MyR]       a[��	ЛR,_��A�"*O

prediction_lossR�?

reg_losslXw<


total_loss��"?


accuracy_1\��>i���]       a[��	��R,_��A�"*O

prediction_loss=
�>

reg_loss`Xw<


total_loss ��>


accuracy_1�z?��t{]       a[��	��R,_��A�"*O

prediction_loss   ?

reg_lossXXw<


total_lossa�?


accuracy_1   ?/�Q�]       a[��	E�R,_��A�"*O

prediction_loss��>

reg_lossMXw<


total_loss�?�>


accuracy_1q=
?r�͗]       a[��	�S,_��A�"*O

prediction_loss�G�>

reg_lossCXw<


total_lossp�>


accuracy_1)\?�q�]       a[��	�;S,_��A�"*O

prediction_loss   ?

reg_loss7Xw<


total_lossa�?


accuracy_1   ?���]       a[��	�`S,_��A�"*O

prediction_loss�G�>

reg_loss.Xw<


total_losso�>


accuracy_1)\?A!�J]       a[��	f�S,_��A�"*O

prediction_loss��>

reg_loss#Xw<


total_loss�?�>


accuracy_1q=
?$yj]       a[��	��S,_��A�"*O

prediction_loss)\?

reg_lossXw<


total_loss�9?


accuracy_1�G�>�LV]       a[��	k�S,_��A�"*O

prediction_loss��>

reg_lossXw<


total_loss�?�>


accuracy_1q=
?��`x]       a[��	��S,_��A�"*O

prediction_loss\��>

reg_lossXw<


total_lossJ�>


accuracy_1R�?���]       a[��	��S,_��A�"*O

prediction_loss��?

reg_loss�Ww<


total_loss�v?


accuracy_1���>v�M^]       a[��	�"T,_��A�"*O

prediction_lossq=
?

reg_loss�Ww<


total_loss�?


accuracy_1��>���]       a[��	�?T,_��A�"*O

prediction_lossq=
?

reg_loss�Ww<


total_loss�?


accuracy_1��>�/CY]       a[��	\T,_��A�"*O

prediction_lossR�?

reg_loss�Ww<


total_loss��"?


accuracy_1\��>����]       a[��	s�T,_��A�"*O

prediction_loss�G�>

reg_loss�Ww<


total_lossl�>


accuracy_1)\?�Hk�]       a[��	��T,_��A�"*O

prediction_loss�z?

reg_loss�Ww<


total_loss@X?


accuracy_1=
�>64�]       a[��	6�T,_��A�"*O

prediction_loss)\?

reg_loss�Ww<


total_loss�9?


accuracy_1�G�>+�c]       a[��	�U,_��A�"*O

prediction_loss
�#?

reg_loss�Ww<


total_lossi�'?


accuracy_1�Q�>�=U]       a[��	TU,_��A�"*O

prediction_loss\��>

reg_loss�Ww<


total_lossJ�>


accuracy_1R�?^��]       a[��	�=U,_��A�"*O

prediction_lossq=
?

reg_loss�Ww<


total_loss�?


accuracy_1��>w��]       a[��	�^U,_��A�"*O

prediction_loss��>

reg_loss�Ww<


total_loss�?�>


accuracy_1q=
?�%B]       a[��	��U,_��A�"*O

prediction_loss�?

reg_loss�Ww<


total_loss�?


accuracy_1���>�C\]       a[��	ܡU,_��A�"*O

prediction_loss�z?

reg_lossWw<


total_loss?X?


accuracy_1=
�>o���]       a[��	f�U,_��A�"*O

prediction_lossq=
?

reg_lossuWw<


total_loss�?


accuracy_1��>l�xx]       a[��	{�U,_��A�"*O

prediction_loss)\?

reg_lossjWw<


total_loss�9?


accuracy_1�G�>��֤]       a[��	_�U,_��A�"*O

prediction_loss)\?

reg_lossaWw<


total_loss�9?


accuracy_1�G�>�/l]       a[��	�V,_��A�"*O

prediction_lossq=
?

reg_lossXWw<


total_loss�?


accuracy_1��>��4t]       a[��	t'V,_��A�"*O

prediction_loss�?

reg_lossMWw<


total_loss�?


accuracy_1���>�y�]       a[��	�BV,_��A�"*O

prediction_loss�z?

reg_lossBWw<


total_loss>X?


accuracy_1=
�>��A ]       a[��	B^V,_��A�"*O

prediction_loss��>

reg_loss8Ww<


total_loss�?�>


accuracy_1q=
?`�q]       a[��	KxV,_��A�"*O

prediction_loss�?

reg_loss.Ww<


total_loss�?


accuracy_1���>~�,]       a[��	��V,_��A�"*O

prediction_loss   ?

reg_loss#Ww<


total_loss]�?


accuracy_1   ?q�,�]       a[��	��V,_��A�"*O

prediction_loss���>

reg_lossWw<


total_lossH}�>


accuracy_1�?O��?]       a[��	��V,_��A�"*O

prediction_loss�G�>

reg_lossWw<


total_lossf�>


accuracy_1)\?�O�]       a[��	
�V,_��A�"*O

prediction_loss��?

reg_lossWw<


total_loss�v?


accuracy_1���>��^q]       a[��	��V,_��A�"*O

prediction_loss��>

reg_loss�Vw<


total_loss�?�>


accuracy_1q=
?�H�>]       a[��	,W,_��A�"*O

prediction_loss
ף>

reg_loss�Vw<


total_loss���>


accuracy_1{.?�W�e]       a[��	F'W,_��A�"*O

prediction_loss�?

reg_loss�Vw<


total_loss�?


accuracy_1���>���^]       a[��	�@W,_��A�"*O

prediction_loss�?

reg_loss�Vw<


total_loss�?


accuracy_1���>+s'U]       a[��	`WW,_��A�"*O

prediction_loss   ?

reg_loss�Vw<


total_loss[�?


accuracy_1   ?8G��]       a[��	�qW,_��A�"*O

prediction_loss�G�>

reg_loss�Vw<


total_lossd�>


accuracy_1)\?�C�]       a[��	��W,_��A�"*O

prediction_loss=
�>

reg_loss�Vw<


total_loss���>


accuracy_1�z?��Z]       a[��	ܠW,_��A�"*O

prediction_loss\��>

reg_loss�Vw<


total_lossJ�>


accuracy_1R�?���]       a[��	V�W,_��A�"*O

prediction_loss���>

reg_loss�Vw<


total_lossD}�>


accuracy_1�?�0Ea]       a[��	�W,_��A�"*O

prediction_loss=
�>

reg_loss�Vw<


total_loss���>


accuracy_1�z?�+ĺ]       a[��	O�W,_��A�"*O

prediction_loss���>

reg_loss�Vw<


total_lossD}�>


accuracy_1�?�P��]       a[��	X,_��A�"*O

prediction_loss�z?

reg_loss�Vw<


total_loss;X?


accuracy_1=
�>h� ]       a[��	�X,_��A�"*O

prediction_loss   ?

reg_loss}Vw<


total_lossZ�?


accuracy_1   ?�W�M]       a[��	�<X,_��A�"*O

prediction_loss���>

reg_losstVw<


total_loss���>


accuracy_1��?@�(]       a[��	�PX,_��A�"*O

prediction_loss   ?

reg_losshVw<


total_lossZ�?


accuracy_1   ?'��]       a[��	�jX,_��A�"*O

prediction_loss���>

reg_loss_Vw<


total_lossB}�>


accuracy_1�?t��]       a[��	@�X,_��A�"*O

prediction_loss�?

reg_lossTVw<


total_loss�?


accuracy_1���>4S�a]       a[��	��X,_��A�"*O

prediction_loss�G�>

reg_lossJVw<


total_loss`�>


accuracy_1)\?���]       a[��	�X,_��A�"*O

prediction_loss�?

reg_loss@Vw<


total_loss�?


accuracy_1���>���>]       a[��	��X,_��A�"*O

prediction_loss��?

reg_loss4Vw<


total_loss�v?


accuracy_1���>�]       a[��	��X,_��A�"*O

prediction_loss�z?

reg_loss+Vw<


total_loss:X?


accuracy_1=
�>EՊ�]       a[��	�Y,_��A�"*O

prediction_loss)\?

reg_loss Vw<


total_loss�9?


accuracy_1�G�>�x��]       a[��	%Y,_��A�"*O

prediction_loss   ?

reg_lossVw<


total_lossX�?


accuracy_1   ?�Í2]       a[��	�5Y,_��A�"*O

prediction_lossR�?

reg_lossVw<


total_loss��"?


accuracy_1\��>�q]       a[��	�RY,_��A�"*O

prediction_loss)\?

reg_lossVw<


total_loss�9?


accuracy_1�G�>7��]       a[��	mY,_��A�"*O

prediction_loss���>

reg_loss�Uw<


total_loss?}�>


accuracy_1�?	�ۘ]       a[��	��Y,_��A�"*O

prediction_loss333?

reg_loss�Uw<


total_loss�7?


accuracy_1���>8�-o]       a[��	M�Y,_��A�"*O

prediction_loss�?

reg_loss�Uw<


total_loss�?


accuracy_1���>X�=]       a[��	иY,_��A�"*O

prediction_lossq=
?

reg_loss�Uw<


total_loss�?


accuracy_1��>���]       a[��	��Y,_��A�"*O

prediction_loss�?

reg_loss�Uw<


total_loss�?


accuracy_1���>��>]       a[��	��Y,_��A�"*O

prediction_loss�z?

reg_loss�Uw<


total_loss8X?


accuracy_1=
�>��r�]       a[��	�Z,_��A�"*O

prediction_loss��?

reg_loss�Uw<


total_loss�v?


accuracy_1���>�/]]       a[��	"Z,_��A�"*O

prediction_loss)\?

reg_loss�Uw<


total_loss�9?


accuracy_1�G�>+��	]       a[��	74Z,_��A�"*O

prediction_loss�?

reg_loss�Uw<


total_loss�?


accuracy_1���>kN.]       a[��	mUZ,_��A�"*O

prediction_loss���>

reg_loss�Uw<


total_lossz��>


accuracy_1��?�_�]       a[��	"oZ,_��A�"*O

prediction_lossq=
?

reg_loss�Uw<


total_loss�?


accuracy_1��>V��\]       a[��	��Z,_��A�"*O

prediction_loss)\?

reg_loss�Uw<


total_loss9?


accuracy_1�G�><�\�]       a[��	��Z,_��A�#*O

prediction_loss���>

reg_loss~Uw<


total_lossy��>


accuracy_1��?7�-[]       a[��	̷Z,_��A�#*O

prediction_loss��?

reg_lossvUw<


total_loss�v?


accuracy_1���>���]       a[��	_�Z,_��A�#*O

prediction_loss���>

reg_lossiUw<


total_lossx��>


accuracy_1��?��ƣ]       a[��	F�Z,_��A�#*O

prediction_loss=
�>

reg_loss_Uw<


total_loss���>


accuracy_1�z?i�k�]       a[��	G[,_��A�#*O

prediction_loss���>

reg_lossUUw<


total_loss:}�>


accuracy_1�?�ռ�]       a[��	�[,_��A�#*O

prediction_loss\��>

reg_lossKUw<


total_lossJ�>


accuracy_1R�?�9��]       a[��		3[,_��A�#*O

prediction_lossq=
?

reg_loss@Uw<


total_loss�?


accuracy_1��>�Id�]       a[��	�H[,_��A�#*O

prediction_loss   ?

reg_loss6Uw<


total_lossU�?


accuracy_1   ?uq�N]       a[��	�c[,_��A�#*O

prediction_loss�z?

reg_loss,Uw<


total_loss6X?


accuracy_1=
�>�jmx]       a[��	�y[,_��A�#*O

prediction_loss   ?

reg_loss"Uw<


total_lossU�?


accuracy_1   ?��~>]       a[��	��[,_��A�#*O

prediction_lossR�?

reg_lossUw<


total_loss��"?


accuracy_1\��>-�\0]       a[��	��[,_��A�#*O

prediction_loss)\?

reg_lossUw<


total_loss}9?


accuracy_1�G�>%T\*]       a[��	��[,_��A�#*O

prediction_lossq=
?

reg_lossUw<


total_loss�?


accuracy_1��>9\.&]       a[��	��[,_��A�#*O

prediction_loss333?

reg_loss�Tw<


total_loss�7?


accuracy_1���>u/O�]       a[��	�[,_��A�#*O

prediction_loss�G�>

reg_loss�Tw<


total_lossU�>


accuracy_1)\?�Mu�]       a[��	�	\,_��A�#*O

prediction_lossq=
?

reg_loss�Tw<


total_loss�?


accuracy_1��>@@��]       a[��	�#\,_��A�#*O

prediction_loss���>

reg_loss�Tw<


total_loss6}�>


accuracy_1�?]��]       a[��	�9\,_��A�#*O

prediction_loss��>

reg_loss�Tw<


total_loss�?�>


accuracy_1q=
?�^5]       a[��		Q\,_��A�#*O

prediction_loss=
�>

reg_loss�Tw<


total_loss���>


accuracy_1�z?��]       a[��	�l\,_��A�#*O

prediction_loss   ?

reg_loss�Tw<


total_lossS�?


accuracy_1   ?/��L]       a[��	�\,_��A�#*O

prediction_loss�z?

reg_loss�Tw<


total_loss4X?


accuracy_1=
�>fr}�]       a[��	t�\,_��A�#*O

prediction_lossq=
?

reg_loss�Tw<


total_loss�?


accuracy_1��>"�ڬ]       a[��	��\,_��A�#*O

prediction_loss   ?

reg_loss�Tw<


total_lossR�?


accuracy_1   ?���f]       a[��	
],_��A�#*O

prediction_loss   ?

reg_loss�Tw<


total_lossR�?


accuracy_1   ?��6�]       a[��	� ],_��A�#*O

prediction_loss�?

reg_losszTw<


total_loss
�?


accuracy_1���>^@��]       a[��	�8],_��A�#*O

prediction_lossq=
?

reg_lossoTw<


total_loss�?


accuracy_1��>���]       a[��	UO],_��A�#*O

prediction_loss�?

reg_lossfTw<


total_loss
�?


accuracy_1���>����]       a[��	8i],_��A�#*O

prediction_loss�Q�>

reg_lossZTw<


total_loss��>


accuracy_1
�#?��B�]       a[��	V�],_��A�#*O

prediction_loss��?

reg_lossPTw<


total_loss�v?


accuracy_1���>�A�]       a[��	��],_��A�#*O

prediction_loss   ?

reg_lossETw<


total_lossQ�?


accuracy_1   ?K�]~]       a[��	y�],_��A�#*O

prediction_loss���>

reg_loss=Tw<


total_loss1}�>


accuracy_1�?�0i]       a[��	��],_��A�#*O

prediction_lossR�?

reg_loss3Tw<


total_loss��"?


accuracy_1\��>�� �]       a[��	�],_��A�#*O

prediction_loss�?

reg_loss(Tw<


total_loss	�?


accuracy_1���>���]       a[��	��],_��A�#*O

prediction_loss���>

reg_lossTw<


total_loss0}�>


accuracy_1�?B�P]       a[��	a^,_��A�#*O

prediction_loss��>

reg_lossTw<


total_loss�?�>


accuracy_1q=
?? ��]       a[��	�1^,_��A�#*O

prediction_loss���>

reg_loss	Tw<


total_loss/}�>


accuracy_1�?��ś]       a[��	�K^,_��A�#*O

prediction_loss��(?

reg_loss�Sw<


total_loss�,?


accuracy_1{�>��n]       a[��	'f^,_��A�#*O

prediction_loss���>

reg_loss�Sw<


total_loss/}�>


accuracy_1�?�TZ]       a[��	�^,_��A�#*O

prediction_loss���>

reg_loss�Sw<


total_loss.}�>


accuracy_1�?�@��]       a[��	A�^,_��A�#*O

prediction_loss�?

reg_loss�Sw<


total_loss�?


accuracy_1���>���]       a[��	�^,_��A�#*O

prediction_loss=
�>

reg_loss�Sw<


total_loss���>


accuracy_1�z?�E��]       a[��	M�^,_��A�#*O

prediction_loss�z?

reg_loss�Sw<


total_loss0X?


accuracy_1=
�>gX$;]       a[��	F	_,_��A�#*O

prediction_loss�?

reg_loss�Sw<


total_loss�?


accuracy_1���>�Ī]       a[��	�>_,_��A�#*O

prediction_loss��>

reg_loss�Sw<


total_loss�?�>


accuracy_1q=
?��L]       a[��	]m_,_��A�#*O

prediction_loss��>

reg_loss�Sw<


total_loss�?�>


accuracy_1q=
?@|��]       a[��	�_,_��A�#*O

prediction_loss���>

reg_loss�Sw<


total_loss,}�>


accuracy_1�?様]       a[��	K:`,_��A�#*O

prediction_lossq=
?

reg_loss�Sw<


total_loss�?


accuracy_1��>�~/d]       a[��	~r`,_��A�#*O

prediction_loss��>

reg_loss�Sw<


total_loss�?�>


accuracy_1q=
?f�Q]       a[��	��`,_��A�#*O

prediction_loss���>

reg_loss�Sw<


total_loss+}�>


accuracy_1�?�Q�]       a[��	��`,_��A�#*O

prediction_loss��>

reg_lossxSw<


total_loss�?�>


accuracy_1q=
?��C]       a[��	��`,_��A�#*O

prediction_loss�?

reg_lossnSw<


total_loss�?


accuracy_1���>A��]       a[��	n�`,_��A�#*O

prediction_lossR�?

reg_lossdSw<


total_loss��"?


accuracy_1\��>���]       a[��	�0a,_��A�#*O

prediction_loss=
�>

reg_lossYSw<


total_loss���>


accuracy_1�z?�:]       a[��	�Na,_��A�#*O

prediction_loss���>

reg_lossMSw<


total_loss)}�>


accuracy_1�?��p]       a[��	+ia,_��A�#*O

prediction_loss
�#?

reg_lossCSw<


total_lossW�'?


accuracy_1�Q�>e-�]       a[��	ۇa,_��A�#*O

prediction_loss   ?

reg_loss:Sw<


total_lossM�?


accuracy_1   ?��E]       a[��	"�a,_��A�#*O

prediction_loss   ?

reg_loss.Sw<


total_lossM�?


accuracy_1   ?���]       a[��	6�a,_��A�#*O

prediction_loss�z?

reg_loss$Sw<


total_loss.X?


accuracy_1=
�>�_�]       a[��	�a,_��A�#*O

prediction_loss)\?

reg_lossSw<


total_lossu9?


accuracy_1�G�>��]       a[��	Wb,_��A�#*O

prediction_loss��?

reg_lossSw<


total_loss�v?


accuracy_1���>3��]       a[��	^.b,_��A�#*O

prediction_loss��?

reg_lossSw<


total_loss�v?


accuracy_1���>Ӧ�]       a[��	4Mb,_��A�#*O

prediction_loss�G�>

reg_loss�Rw<


total_lossF�>


accuracy_1)\?6&�	]       a[��	geb,_��A�#*O

prediction_loss��>

reg_loss�Rw<


total_loss�?�>


accuracy_1q=
?�*�]       a[��	�|b,_��A�#*O

prediction_loss�z?

reg_loss�Rw<


total_loss-X?


accuracy_1=
�>j��]       a[��	Q�b,_��A�#*O

prediction_loss=
�>

reg_loss�Rw<


total_loss���>


accuracy_1�z?G�r�]       a[��	��b,_��A�#*O

prediction_loss���>

reg_loss�Rw<


total_loss1T�>


accuracy_1333?�}a�]       a[��	��b,_��A�#*O

prediction_loss)\?

reg_loss�Rw<


total_losst9?


accuracy_1�G�>�\�&]       a[��	\�b,_��A�#*O

prediction_loss��>

reg_loss�Rw<


total_loss�?�>


accuracy_1q=
?��V�]       a[��	�c,_��A�#*O

prediction_loss)\?

reg_loss�Rw<


total_losst9?


accuracy_1�G�>sl]       a[��	�;c,_��A�#*O

prediction_loss��?

reg_loss�Rw<


total_loss�v?


accuracy_1���>�fҌ]       a[��	!Xc,_��A�#*O

prediction_loss�G�>

reg_loss�Rw<


total_lossC�>


accuracy_1)\?L]       a[��	tc,_��A�#*O

prediction_loss���>

reg_loss�Rw<


total_lossb��>


accuracy_1��?UK]       a[��	3�c,_��A�#*O

prediction_loss�?

reg_loss�Rw<


total_loss�?


accuracy_1���>�~�]       a[��	9�c,_��A�#*O

prediction_loss��?

reg_loss|Rw<


total_loss�v?


accuracy_1���>ȧ`]       a[��	��c,_��A�#*O

prediction_lossq=
?

reg_lossrRw<


total_loss�?


accuracy_1��>؆�]       a[��	��c,_��A�#*O

prediction_loss   ?

reg_losshRw<


total_lossJ�?


accuracy_1   ?�M#^]       a[��	l
d,_��A�#*O

prediction_loss�z?

reg_loss_Rw<


total_loss*X?


accuracy_1=
�>g��]       a[��	�+d,_��A�#*O

prediction_loss�z?

reg_lossWRw<


total_loss*X?


accuracy_1=
�>�k�,]       a[��	�Md,_��A�#*O

prediction_loss��>

reg_lossKRw<


total_loss�?�>


accuracy_1q=
?Iq��]       a[��	bid,_��A�#*O

prediction_loss)\?

reg_lossBRw<


total_lossr9?


accuracy_1�G�>���]       a[��	b�d,_��A�#*O

prediction_lossR�?

reg_loss6Rw<


total_loss��"?


accuracy_1\��>��]       a[��	�d,_��A�#*O

prediction_loss���>

reg_loss,Rw<


total_loss }�>


accuracy_1�?>���]       a[��	��d,_��A�#*O

prediction_lossq=
?

reg_loss!Rw<


total_loss�?


accuracy_1��>�+Ӛ]       a[��	��d,_��A�#*O

prediction_loss
ף>

reg_lossRw<


total_loss���>


accuracy_1{.?�zu]       a[��	��d,_��A�#*O

prediction_loss�G�>

reg_lossRw<


total_loss>�>


accuracy_1)\?fZ�]       a[��	�e,_��A�#*O

prediction_loss   ?

reg_lossRw<


total_lossH�?


accuracy_1   ?�UEp]       a[��	�4e,_��A�#*O

prediction_loss   ?

reg_loss�Qw<


total_lossH�?


accuracy_1   ?Nd�]       a[��		Qe,_��A�#*O

prediction_loss���>

reg_loss�Qw<


total_loss}�>


accuracy_1�?1�M�]       a[��	oe,_��A�#*O

prediction_lossR�?

reg_loss�Qw<


total_loss��"?


accuracy_1\��>���F]       a[��	#�e,_��A�#*O

prediction_loss   ?

reg_loss�Qw<


total_lossG�?


accuracy_1   ?,�]       a[��	=�e,_��A�#*O

prediction_loss)\?

reg_loss�Qw<


total_lossp9?


accuracy_1�G�>`#�S]       a[��	��e,_��A�#*O

prediction_loss   ?

reg_loss�Qw<


total_lossG�?


accuracy_1   ?]*�=]       a[��	{�e,_��A�#*O

prediction_loss)\?

reg_loss�Qw<


total_lossp9?


accuracy_1�G�>��*�]       a[��	b�e,_��A�#*O

prediction_loss
ף>

reg_loss�Qw<


total_loss���>


accuracy_1{.?z;Y]       a[��		f,_��A�#*O

prediction_loss���>

reg_loss�Qw<


total_loss}�>


accuracy_1�?%�]       a[��	d;f,_��A�#*O

prediction_lossq=
?

reg_loss�Qw<


total_loss�?


accuracy_1��>Ӗ;\]       a[��	�Wf,_��A�#*O

prediction_loss�?

reg_loss�Qw<


total_loss��?


accuracy_1���>$��?]       a[��	�nf,_��A�#*O

prediction_loss���>

reg_loss�Qw<


total_lossY��>


accuracy_1��?Z0?�]       a[��	�f,_��A�#*O

prediction_loss�Q�>

reg_loss�Qw<


total_lossx�>


accuracy_1
�#?�9�]       a[��	�f,_��A�#*O

prediction_loss�z?

reg_lossvQw<


total_loss'X?


accuracy_1=
�>��X�]       a[��	 �f,_��A�#*O

prediction_loss)\?

reg_losslQw<


total_losso9?


accuracy_1�G�>Wb�&]       a[��	��f,_��A�#*O

prediction_loss�?

reg_losscQw<


total_loss��?


accuracy_1���>���3]       a[��	��f,_��A�#*O

prediction_loss���>

reg_lossXQw<


total_loss}�>


accuracy_1�?Ijo�]       a[��	^g,_��A�#*O

prediction_loss��?

reg_lossNQw<


total_loss�v?


accuracy_1���>�vY�]       a[��	�'g,_��A�#*O

prediction_loss�G�>

reg_lossBQw<


total_loss8�>


accuracy_1)\?(=�]       a[��	�Fg,_��A�#*O

prediction_lossq=
?

reg_loss9Qw<


total_loss�?


accuracy_1��>B�z]       a[��	Rbg,_��A�#*O

prediction_loss���>

reg_loss/Qw<


total_lossV��>


accuracy_1��?��y]       a[��	�{g,_��A�#*O

prediction_loss�Q�>

reg_loss%Qw<


total_lossu�>


accuracy_1
�#?�j��]       a[��	W�g,_��A�#*O

prediction_loss��>

reg_lossQw<


total_loss�?�>


accuracy_1q=
?�{r�]       a[��	)�g,_��A�#*O

prediction_loss\��>

reg_lossQw<


total_loss�I�>


accuracy_1R�?jD?C]       a[��	_�g,_��A�#*O

prediction_loss���>

reg_lossQw<


total_lossU��>


accuracy_1��?v�t]       a[��	��g,_��A�#*O

prediction_lossR�?

reg_loss�Pw<


total_loss��"?


accuracy_1\��>�K*�]       a[��	�h,_��A�#*O

prediction_loss{�>

reg_loss�Pw<


total_lossϵ>


accuracy_1��(?� ��]       a[��	Th,_��A�#*O

prediction_loss��>

reg_loss�Pw<


total_loss�?�>


accuracy_1q=
?����]       a[��	/5h,_��A�#*O

prediction_loss=
�>

reg_loss�Pw<


total_loss���>


accuracy_1�z?8��]       a[��	KVh,_��A�#*O

prediction_loss�z?

reg_loss�Pw<


total_loss$X?


accuracy_1=
�>���Y]       a[��	�nh,_��A�#*O

prediction_loss�z?

reg_loss�Pw<


total_loss$X?


accuracy_1=
�>�|}]       a[��	��h,_��A�#*O

prediction_loss=
�>

reg_loss�Pw<


total_loss���>


accuracy_1�z?�B�]       a[��	w�h,_��A�#*O

prediction_loss���>

reg_loss�Pw<


total_loss}�>


accuracy_1�?x���]       a[��	��h,_��A�#*O

prediction_lossq=
?

reg_loss�Pw<


total_loss�?


accuracy_1��>PC�]       a[��	��h,_��A�#*O

prediction_loss�?

reg_loss�Pw<


total_loss��?


accuracy_1���>����]       a[��	/i,_��A�#*O

prediction_loss�G�>

reg_loss�Pw<


total_loss2�>


accuracy_1)\?ܻ��]       a[��	�Pi,_��A�#*O

prediction_loss�?

reg_loss�Pw<


total_loss��?


accuracy_1���>�0�]       a[��	�ni,_��A�#*O

prediction_lossq=
?

reg_lossvPw<


total_loss�?


accuracy_1��>ܔ�]       a[��	�i,_��A�#*O

prediction_loss�G�>

reg_losslPw<


total_loss1�>


accuracy_1)\?� J�]       a[��	��i,_��A�#*O

prediction_loss�G�>

reg_lossaPw<


total_loss1�>


accuracy_1)\?"���]       a[��	��i,_��A�#*O

prediction_lossq=
?

reg_lossVPw<


total_loss�?


accuracy_1��>��]       a[��	�i,_��A�#*O

prediction_loss�z?

reg_lossMPw<


total_loss"X?


accuracy_1=
�>?���]       a[��	�i,_��A�$*O

prediction_loss���>

reg_lossAPw<


total_loss}�>


accuracy_1�?B�2�]       a[��	4j,_��A�$*O

prediction_loss��>

reg_loss5Pw<


total_loss�?�>


accuracy_1q=
?�gf]       a[��	�+j,_��A�$*O

prediction_loss�z?

reg_loss*Pw<


total_loss"X?


accuracy_1=
�>\:�]       a[��	Ej,_��A�$*O

prediction_loss)\?

reg_loss!Pw<


total_lossj9?


accuracy_1�G�>&+G]       a[��	cj,_��A�$*O

prediction_loss��>

reg_lossPw<


total_loss�?�>


accuracy_1q=
?[�]       a[��	�|j,_��A�$*O

prediction_loss�G�>

reg_lossPw<


total_loss.�>


accuracy_1)\?�}g]       a[��	��j,_��A�$*O

prediction_loss���>

reg_lossPw<


total_loss}�>


accuracy_1�?c)�]       a[��	��j,_��A�$*O

prediction_loss���>

reg_loss�Ow<


total_loss}�>


accuracy_1�?�Y��]       a[��	�j,_��A�$*O

prediction_loss)\?

reg_loss�Ow<


total_lossi9?


accuracy_1�G�>��҉]       a[��	]�j,_��A�$*O

prediction_loss��?

reg_loss�Ow<


total_loss�v?


accuracy_1���>���]       a[��	��j,_��A�$*O

prediction_loss���>

reg_loss�Ow<


total_loss}�>


accuracy_1�?^�w]       a[��	�k,_��A�$*O

prediction_loss��>

reg_loss�Ow<


total_loss�?�>


accuracy_1q=
?(��]       a[��	7k,_��A�$*O

prediction_loss��(?

reg_loss�Ow<


total_loss�,?


accuracy_1{�>d}O\]       a[��	nnk,_��A�$*O

prediction_loss�?

reg_loss�Ow<


total_loss��?


accuracy_1���>&�]       a[��	��k,_��A�$*O

prediction_loss{�>

reg_loss�Ow<


total_loss�ε>


accuracy_1��(?rk]       a[��	��k,_��A�$*O

prediction_loss{.?

reg_loss�Ow<


total_loss��1?


accuracy_1
ף>4^��]       a[��	�l,_��A�$*O

prediction_loss��>

reg_loss�Ow<


total_loss�?�>


accuracy_1q=
?Qɟ�]       a[��	I/l,_��A�$*O

prediction_loss���>

reg_loss�Ow<


total_loss}�>


accuracy_1�?���R]       a[��	�Kl,_��A�$*O

prediction_loss��(?

reg_loss�Ow<


total_loss�,?


accuracy_1{�>\uu]       a[��	"ol,_��A�$*O

prediction_loss���>

reg_loss~Ow<


total_loss}�>


accuracy_1�?>q+.]       a[��	ӈl,_��A�$*O

prediction_loss�G�>

reg_losstOw<


total_loss*�>


accuracy_1)\?�6u]       a[��	��l,_��A�$*O

prediction_loss\��>

reg_losshOw<


total_loss�I�>


accuracy_1R�?�v��]       a[��	7�l,_��A�$*O

prediction_loss�G�>

reg_loss_Ow<


total_loss)�>


accuracy_1)\?�8�w]       a[��	��l,_��A�$*O

prediction_loss�?

reg_lossUOw<


total_loss��?


accuracy_1���>ڿ�]       a[��	%m,_��A�$*O

prediction_loss=
�>

reg_lossKOw<


total_loss���>


accuracy_1�z?S|�]       a[��	�#m,_��A�$*O

prediction_loss���>

reg_loss@Ow<


total_lossG��>


accuracy_1��?��CJ]       a[��	�@m,_��A�$*O

prediction_loss)\?

reg_loss7Ow<


total_lossf9?


accuracy_1�G�>I��+]       a[��	�_m,_��A�$*O

prediction_loss�z?

reg_loss-Ow<


total_lossX?


accuracy_1=
�>��o�]       a[��	��m,_��A�$*O

prediction_lossq=
?

reg_loss#Ow<


total_loss�?


accuracy_1��>H�]       a[��	��m,_��A�$*O

prediction_loss
�#?

reg_lossOw<


total_lossF�'?


accuracy_1�Q�>Y���]       a[��	��m,_��A�$*O

prediction_loss���>

reg_lossOw<


total_lossE��>


accuracy_1��?����]       a[��	��m,_��A�$*O

prediction_loss)\?

reg_lossOw<


total_losse9?


accuracy_1�G�>��;�]       a[��	��m,_��A�$*O

prediction_loss=
�>

reg_loss�Nw<


total_loss���>


accuracy_1�z?�6]       a[��	n,_��A�$*O

prediction_loss��>

reg_loss�Nw<


total_loss�?�>


accuracy_1q=
?��]       a[��	A.n,_��A�$*O

prediction_loss�?

reg_loss�Nw<


total_loss��?


accuracy_1���>'X��]       a[��	�Gn,_��A�$*O

prediction_loss�G�>

reg_loss�Nw<


total_loss%�>


accuracy_1)\?�;�_]       a[��	�`n,_��A�$*O

prediction_loss{�>

reg_loss�Nw<


total_loss�ε>


accuracy_1��(?�S��]       a[��	|�n,_��A�$*O

prediction_loss��>

reg_loss�Nw<


total_loss�?�>


accuracy_1q=
?j��]       a[��	̘n,_��A�$*O

prediction_loss)\?

reg_loss�Nw<


total_lossd9?


accuracy_1�G�>럻�]       a[��	մn,_��A�$*O

prediction_loss��>

reg_loss�Nw<


total_loss�?�>


accuracy_1q=
?
Fuz]       a[��	��n,_��A�$*O

prediction_lossq=
?

reg_loss�Nw<


total_loss�?


accuracy_1��>�n��]       a[��	�n,_��A�$*O

prediction_loss���>

reg_loss�Nw<


total_loss}�>


accuracy_1�?N�o]       a[��	�o,_��A�$*O

prediction_loss   ?

reg_loss�Nw<


total_loss:�?


accuracy_1   ?6���]       a[��	�&o,_��A�$*O

prediction_loss�z?

reg_loss�Nw<


total_lossX?


accuracy_1=
�>!8O�]       a[��	�Ao,_��A�$*O

prediction_loss��(?

reg_loss|Nw<


total_loss��,?


accuracy_1{�>�	�\]       a[��	�[o,_��A�$*O

prediction_lossq=
?

reg_lossrNw<


total_loss�?


accuracy_1��>�2
�]       a[��	�xo,_��A�$*O

prediction_loss   ?

reg_lossiNw<


total_loss:�?


accuracy_1   ?���F]       a[��	a�o,_��A�$*O

prediction_loss�?

reg_loss_Nw<


total_loss��?


accuracy_1���>_���]       a[��	��o,_��A�$*O

prediction_loss��(?

reg_lossTNw<


total_loss��,?


accuracy_1{�>;֣]       a[��	�p,_��A�$*O

prediction_loss=
�>

reg_lossJNw<


total_loss���>


accuracy_1�z?��c]       a[��	IGp,_��A�$*O

prediction_loss�z?

reg_loss>Nw<


total_lossX?


accuracy_1=
�>��X]       a[��	�xp,_��A�$*O

prediction_loss���>

reg_loss6Nw<


total_lossT�>


accuracy_1333?0˦]       a[��	��p,_��A�$*O

prediction_loss)\?

reg_loss*Nw<


total_lossb9?


accuracy_1�G�>�v-z]       a[��	ճp,_��A�$*O

prediction_loss�z?

reg_loss!Nw<


total_lossX?


accuracy_1=
�>jaN]       a[��	�p,_��A�$*O

prediction_loss   ?

reg_lossNw<


total_loss8�?


accuracy_1   ?B�a�]       a[��	��p,_��A�$*O

prediction_lossR�?

reg_lossNw<


total_loss��"?


accuracy_1\��>i�R�]       a[��	�q,_��A�$*O

prediction_loss   ?

reg_lossNw<


total_loss8�?


accuracy_1   ?�d��]       a[��	�Eq,_��A�$*O

prediction_loss�G�>

reg_loss�Mw<


total_loss�>


accuracy_1)\?��]       a[��	�bq,_��A�$*O

prediction_lossR�?

reg_loss�Mw<


total_loss��"?


accuracy_1\��>�N �]       a[��	�q,_��A�$*O

prediction_loss   ?

reg_loss�Mw<


total_loss8�?


accuracy_1   ?��TQ]       a[��	Z�q,_��A�$*O

prediction_loss   ?

reg_loss�Mw<


total_loss7�?


accuracy_1   ?˳�]       a[��	n�q,_��A�$*O

prediction_loss��>

reg_loss�Mw<


total_loss�?�>


accuracy_1q=
?�?A]       a[��	D�q,_��A�$*O

prediction_loss=
�>

reg_loss�Mw<


total_loss���>


accuracy_1�z?S�(]       a[��	#�q,_��A�$*O

prediction_loss���>

reg_loss�Mw<


total_loss�|�>


accuracy_1�?��c]       a[��	�r,_��A�$*O

prediction_loss)\?

reg_loss�Mw<


total_loss`9?


accuracy_1�G�>u(+{]       a[��	�2r,_��A�$*O

prediction_lossq=
?

reg_loss�Mw<


total_loss�?


accuracy_1��>i=x�]       a[��	�Lr,_��A�$*O

prediction_loss���>

reg_loss�Mw<


total_loss:��>


accuracy_1��?���]       a[��	�hr,_��A�$*O

prediction_loss�?

reg_loss�Mw<


total_loss��?


accuracy_1���>d1�W]       a[��	��r,_��A�$*O

prediction_loss   ?

reg_loss�Mw<


total_loss6�?


accuracy_1   ?i�]       a[��	P�r,_��A�$*O

prediction_loss   ?

reg_loss{Mw<


total_loss6�?


accuracy_1   ?����]       a[��	��r,_��A�$*O

prediction_loss�?

reg_lossoMw<


total_loss��?


accuracy_1���>$�]       a[��	a�r,_��A�$*O

prediction_lossq=
?

reg_losseMw<


total_loss�?


accuracy_1��>�W��]       a[��	s,_��A�$*O

prediction_loss���>

reg_loss[Mw<


total_loss�|�>


accuracy_1�?B���]       a[��	5s,_��A�$*O

prediction_lossq=
?

reg_lossRMw<


total_loss�?


accuracy_1��>w]�]       a[��	BZs,_��A�$*O

prediction_loss=
�>

reg_lossEMw<


total_loss���>


accuracy_1�z?N��]       a[��	d�s,_��A�$*O

prediction_loss�?

reg_loss;Mw<


total_loss��?


accuracy_1���>��]       a[��	=�s,_��A�$*O

prediction_loss   ?

reg_loss1Mw<


total_loss5�?


accuracy_1   ?��]       a[��	
�s,_��A�$*O

prediction_loss\��>

reg_loss'Mw<


total_loss�I�>


accuracy_1R�?����]       a[��	t,_��A�$*O

prediction_loss�G�>

reg_lossMw<


total_loss�>


accuracy_1)\?��3Q]       a[��	�7t,_��A�$*O

prediction_loss)\?

reg_lossMw<


total_loss]9?


accuracy_1�G�>����]       a[��	<�t,_��A�$*O

prediction_loss\��>

reg_loss
Mw<


total_loss�I�>


accuracy_1R�?�n��]       a[��	��t,_��A�$*O

prediction_loss)\?

reg_loss�Lw<


total_loss]9?


accuracy_1�G�>��cb]       a[��	��t,_��A�$*O

prediction_loss�?

reg_loss�Lw<


total_loss��?


accuracy_1���>W$@�]       a[��	)�t,_��A�$*O

prediction_lossq=
?

reg_loss�Lw<


total_loss�?


accuracy_1��>����]       a[��	pu,_��A�$*O

prediction_loss�G�>

reg_loss�Lw<


total_loss�>


accuracy_1)\?�Cg]       a[��	�&u,_��A�$*O

prediction_loss��>

reg_loss�Lw<


total_loss�?�>


accuracy_1q=
?O`��]       a[��	9Bu,_��A�$*O

prediction_loss�G�>

reg_loss�Lw<


total_loss�>


accuracy_1)\?�i/]       a[��	!Xu,_��A�$*O

prediction_loss���>

reg_loss�Lw<


total_loss�|�>


accuracy_1�?m�w�]       a[��	dwu,_��A�$*O

prediction_loss��?

reg_loss�Lw<


total_loss�v?


accuracy_1���>��]       a[��	ݗu,_��A�$*O

prediction_loss���>

reg_loss�Lw<


total_loss2��>


accuracy_1��?d��u]       a[��	ܼu,_��A�$*O

prediction_loss���>

reg_loss�Lw<


total_loss2��>


accuracy_1��?�P��]       a[��	
�u,_��A�$*O

prediction_loss=
�>

reg_loss�Lw<


total_loss���>


accuracy_1�z?RR?�]       a[��	��u,_��A�$*O

prediction_loss���>

reg_loss�Lw<


total_loss1��>


accuracy_1��?y��]       a[��	Tv,_��A�$*O

prediction_lossR�?

reg_lossxLw<


total_loss��"?


accuracy_1\��>�X�]       a[��	lv,_��A�$*O

prediction_loss���>

reg_lossmLw<


total_loss0��>


accuracy_1��?Fs�Q]       a[��	�v,_��A�$*O

prediction_loss�z?

reg_lossbLw<


total_lossX?


accuracy_1=
�>*� ]       a[��	�v,_��A�$*O

prediction_loss��?

reg_lossYLw<


total_loss�v?


accuracy_1���>��>]       a[��	��v,_��A�$*O

prediction_lossq=
?

reg_lossOLw<


total_loss�?


accuracy_1��>)��f]       a[��	w�v,_��A�$*O

prediction_loss�?

reg_lossDLw<


total_loss��?


accuracy_1���>
�pD]       a[��	��v,_��A�$*O

prediction_loss���>

reg_loss9Lw<


total_loss/��>


accuracy_1��?��0]       a[��	�w,_��A�$*O

prediction_loss��>

reg_loss0Lw<


total_loss�?�>


accuracy_1q=
?�]       a[��	,.w,_��A�$*O

prediction_loss���>

reg_loss(Lw<


total_loss.��>


accuracy_1��?�յ]       a[��	Gw,_��A�$*O

prediction_loss��?

reg_lossLw<


total_loss�v?


accuracy_1���>���]       a[��	�fw,_��A�$*O

prediction_loss�Q�>

reg_lossLw<


total_lossM�>


accuracy_1
�#?D]O]       a[��	�w,_��A�$*O

prediction_loss�G�>

reg_lossLw<


total_loss�>


accuracy_1)\?Y)�]       a[��	b�w,_��A�$*O

prediction_loss�G�>

reg_loss�Kw<


total_loss�>


accuracy_1)\?/��&]       a[��	�w,_��A�$*O

prediction_loss��(?

reg_loss�Kw<


total_loss��,?


accuracy_1{�>�P��]       a[��	,�w,_��A�$*O

prediction_loss��?

reg_loss�Kw<


total_loss�v?


accuracy_1���>�Po]       a[��	��w,_��A�$*O

prediction_loss��?

reg_loss�Kw<


total_loss�v?


accuracy_1���>M�b�]       a[��	>x,_��A�$*O

prediction_loss)\?

reg_loss�Kw<


total_lossX9?


accuracy_1�G�>P���]       a[��	h!x,_��A�$*O

prediction_loss)\�>

reg_loss�Kw<


total_loss��>


accuracy_1�Q8?�;S]       a[��	�Ex,_��A�$*O

prediction_lossq=
?

reg_loss�Kw<


total_loss�?


accuracy_1��>Bm1]       a[��	�cx,_��A�$*O

prediction_loss�?

reg_loss�Kw<


total_loss��?


accuracy_1���>�(��]       a[��	�x,_��A�$*O

prediction_loss   ?

reg_loss�Kw<


total_loss/�?


accuracy_1   ?�dp�]       a[��	��x,_��A�$*O

prediction_lossR�?

reg_loss�Kw<


total_loss��"?


accuracy_1\��>�z�%]       a[��	k�x,_��A�$*O

prediction_loss�z?

reg_loss�Kw<


total_lossX?


accuracy_1=
�>�Ɓu]       a[��	� y,_��A�$*O

prediction_loss�G�>

reg_loss�Kw<


total_loss
�>


accuracy_1)\?he�]       a[��	SAy,_��A�$*O

prediction_loss   ?

reg_loss�Kw<


total_loss.�?


accuracy_1   ?�|W]       a[��	-\y,_��A�$*O

prediction_loss)\?

reg_lossvKw<


total_lossW9?


accuracy_1�G�>�^3L]       a[��	Cuy,_��A�$*O

prediction_loss=
�>

reg_losskKw<


total_loss���>


accuracy_1�z?�$��]       a[��	�y,_��A�$*O

prediction_lossR�?

reg_losscKw<


total_loss��"?


accuracy_1\��>=�_&]       a[��	%�y,_��A�$*O

prediction_loss�?

reg_lossXKw<


total_loss��?


accuracy_1���>����]       a[��	��y,_��A�$*O

prediction_loss���>

reg_lossMKw<


total_loss'��>


accuracy_1��?���n]       a[��	��y,_��A�$*O

prediction_loss��>

reg_lossCKw<


total_lossy?�>


accuracy_1q=
?j�l]       a[��	�y,_��A�$*O

prediction_loss��?

reg_loss:Kw<


total_loss�v?


accuracy_1���>��O�]       a[��	z,_��A�$*O

prediction_loss   ?

reg_loss0Kw<


total_loss-�?


accuracy_1   ?��;�]       a[��	I*z,_��A�$*O

prediction_loss�z?

reg_loss$Kw<


total_lossX?


accuracy_1=
�>��_�]       a[��	�Dz,_��A�$*O

prediction_lossq=
?

reg_lossKw<


total_loss�?


accuracy_1��>�0��]       a[��	�_z,_��A�%*O

prediction_lossq=
?

reg_lossKw<


total_loss�?


accuracy_1��>;���]       a[��	�wz,_��A�%*O

prediction_loss   ?

reg_lossKw<


total_loss,�?


accuracy_1   ?��,r]       a[��	��z,_��A�%*O

prediction_loss   ?

reg_loss�Jw<


total_loss,�?


accuracy_1   ?
Nl]       a[��	2�z,_��A�%*O

prediction_loss��>

reg_loss�Jw<


total_lossw?�>


accuracy_1q=
?4.K~]       a[��	e�z,_��A�%*O

prediction_loss=
�>

reg_loss�Jw<


total_loss���>


accuracy_1�z?���S]       a[��	��z,_��A�%*O

prediction_loss��>

reg_loss�Jw<


total_lossv?�>


accuracy_1q=
?�׌]       a[��	f�z,_��A�%*O

prediction_loss)\?

reg_loss�Jw<


total_lossT9?


accuracy_1�G�>���T]       a[��	A{,_��A�%*O

prediction_loss��>

reg_loss�Jw<


total_lossu?�>


accuracy_1q=
?�lu�]       a[��	�({,_��A�%*O

prediction_loss=
�>

reg_loss�Jw<


total_loss���>


accuracy_1�z?���=]       a[��	B{,_��A�%*O

prediction_loss���>

reg_loss�Jw<


total_loss�|�>


accuracy_1�?`L�s]       a[��	�Y{,_��A�%*O

prediction_loss333?

reg_loss�Jw<


total_loss^7?


accuracy_1���>M{��]       a[��	�r{,_��A�%*O

prediction_loss�z?

reg_loss�Jw<


total_lossX?


accuracy_1=
�>$�m]       a[��	Ҋ{,_��A�%*O

prediction_loss�G�>

reg_loss�Jw<


total_loss�>


accuracy_1)\?t�i�]       a[��	Q�{,_��A�%*O

prediction_lossq=
?

reg_loss�Jw<


total_loss�?


accuracy_1��>@�q]       a[��	��{,_��A�%*O

prediction_loss�G�>

reg_loss�Jw<


total_loss�>


accuracy_1)\?� r]       a[��	_�{,_��A�%*O

prediction_loss\��>

reg_lossyJw<


total_loss�I�>


accuracy_1R�?�lѳ]       a[��	��{,_��A�%*O

prediction_loss���>

reg_lossmJw<


total_loss ��>


accuracy_1��?J̜A]       a[��	�|,_��A�%*O

prediction_loss   ?

reg_lossbJw<


total_loss*�?


accuracy_1   ?���]       a[��	4|,_��A�%*O

prediction_loss��>

reg_lossXJw<


total_lossr?�>


accuracy_1q=
?��1]       a[��	"U|,_��A�%*O

prediction_lossq=
?

reg_lossNJw<


total_loss�?


accuracy_1��>4���]       a[��	�i|,_��A�%*O

prediction_loss�z?

reg_lossCJw<


total_loss
X?


accuracy_1=
�>��m]       a[��	��|,_��A�%*O

prediction_lossq=
?

reg_loss8Jw<


total_loss�?


accuracy_1��>�d6[]       a[��	צ|,_��A�%*O

prediction_loss�z?

reg_loss.Jw<


total_loss
X?


accuracy_1=
�>��^�]       a[��	�|,_��A�%*O

prediction_loss��>

reg_loss%Jw<


total_lossp?�>


accuracy_1q=
?e��]       a[��	u},_��A�%*O

prediction_loss��?

reg_lossJw<


total_loss�v?


accuracy_1���>�NV�]       a[��	2},_��A�%*O

prediction_loss=
�>

reg_lossJw<


total_loss���>


accuracy_1�z?���]       a[��	6},_��A�%*O

prediction_loss\��>

reg_lossJw<


total_loss�I�>


accuracy_1R�?�įl]       a[��	�L},_��A�%*O

prediction_loss\��>

reg_loss�Iw<


total_loss�I�>


accuracy_1R�?�>��]       a[��	9c},_��A�%*O

prediction_loss���>

reg_loss�Iw<


total_loss�|�>


accuracy_1�?�&)]       a[��	��},_��A�%*O

prediction_loss��>

reg_loss�Iw<


total_lossn?�>


accuracy_1q=
?��]       a[��	��},_��A�%*O

prediction_loss�z?

reg_loss�Iw<


total_lossX?


accuracy_1=
�>�'��]       a[��	,�},_��A�%*O

prediction_loss
ף>

reg_loss�Iw<


total_lossY��>


accuracy_1{.?m�3�]       a[��	��},_��A�%*O

prediction_loss�G�>

reg_loss�Iw<


total_loss��>


accuracy_1)\?GE�]       a[��	�~,_��A�%*O

prediction_loss��>

reg_loss�Iw<


total_lossm?�>


accuracy_1q=
?I��]       a[��	J'~,_��A�%*O

prediction_loss=
�>

reg_loss�Iw<


total_loss���>


accuracy_1�z?�!3�]       a[��	A~,_��A�%*O

prediction_loss�?

reg_loss�Iw<


total_loss��?


accuracy_1���>�ϝB]       a[��	>\~,_��A�%*O

prediction_loss�?

reg_loss�Iw<


total_loss��?


accuracy_1���>^��D]       a[��	.s~,_��A�%*O

prediction_loss   ?

reg_loss�Iw<


total_loss&�?


accuracy_1   ?�,{m]       a[��	m�~,_��A�%*O

prediction_loss���>

reg_loss�Iw<


total_loss�|�>


accuracy_1�?ցB,]       a[��	��~,_��A�%*O

prediction_loss��>

reg_loss�Iw<


total_lossk?�>


accuracy_1q=
?��]       a[��	��~,_��A�%*O

prediction_loss���>

reg_losstIw<


total_loss�|�>


accuracy_1�?��pM]       a[��	]�~,_��A�%*O

prediction_loss�z?

reg_lossjIw<


total_lossX?


accuracy_1=
�>�ZQ}]       a[��	�,_��A�%*O

prediction_loss   ?

reg_loss`Iw<


total_loss&�?


accuracy_1   ?̆�]       a[��	�%,_��A�%*O

prediction_loss��>

reg_lossVIw<


total_lossj?�>


accuracy_1q=
?*���]       a[��	1A,_��A�%*O

prediction_loss=
�>

reg_lossMIw<


total_loss���>


accuracy_1�z?I~�]       a[��	>Z,_��A�%*O

prediction_loss�z?

reg_loss@Iw<


total_lossX?


accuracy_1=
�>�!hG]       a[��	Wx,_��A�%*O

prediction_lossq=
?

reg_loss6Iw<


total_loss�?


accuracy_1��>'���]       a[��	6�,_��A�%*O

prediction_loss\��>

reg_loss,Iw<


total_loss�I�>


accuracy_1R�?�?t]       a[��	�,_��A�%*O

prediction_loss)\?

reg_loss"Iw<


total_lossN9?


accuracy_1�G�>���]       a[��	P�,_��A�%*O

prediction_loss��?

reg_lossIw<


total_loss�v?


accuracy_1���>	��g]       a[��	��,_��A�%*O

prediction_loss�z?

reg_lossIw<


total_lossX?


accuracy_1=
�>���n]       a[��	��,_��A�%*O

prediction_loss�G�>

reg_lossIw<


total_loss��>


accuracy_1)\?$�]�]       a[��	��,_��A�%*O

prediction_loss���>

reg_loss�Hw<


total_loss�|�>


accuracy_1�?犰�]       a[��	�3�,_��A�%*O

prediction_loss���>

reg_loss�Hw<


total_loss�|�>


accuracy_1�?���]       a[��	8J�,_��A�%*O

prediction_loss���>

reg_loss�Hw<


total_loss�|�>


accuracy_1�?h��]       a[��	e�,_��A�%*O

prediction_loss   ?

reg_loss�Hw<


total_loss#�?


accuracy_1   ?���]]       a[��	��,_��A�%*O

prediction_loss   ?

reg_loss�Hw<


total_loss#�?


accuracy_1   ?��]       a[��	0��,_��A�%*O

prediction_lossq=
?

reg_loss�Hw<


total_loss�?


accuracy_1��>���]       a[��	ú�,_��A�%*O

prediction_lossq=
?

reg_loss�Hw<


total_loss�?


accuracy_1��>�r��]       a[��	Ԁ,_��A�%*O

prediction_loss�z?

reg_loss�Hw<


total_lossX?


accuracy_1=
�>���]       a[��	��,_��A�%*O

prediction_lossR�?

reg_loss�Hw<


total_lossu�"?


accuracy_1\��>�+]       a[��	��,_��A�%*O

prediction_loss�?

reg_loss�Hw<


total_loss��?


accuracy_1���>���$]       a[��	 �,_��A�%*O

prediction_loss�?

reg_loss�Hw<


total_loss��?


accuracy_1���>Hj��]       a[��	q9�,_��A�%*O

prediction_loss�G�>

reg_loss�Hw<


total_loss��>


accuracy_1)\?�,��]       a[��	PQ�,_��A�%*O

prediction_loss{�>

reg_loss|Hw<


total_loss�ε>


accuracy_1��(?b#��]       a[��	���,_��A�%*O

prediction_loss   ?

reg_lossgHw<


total_loss"�?


accuracy_1   ?��E]       a[��	�Á,_��A�%*O

prediction_lossq=
?

reg_loss\Hw<


total_loss�?


accuracy_1��>����]       a[��	�݁,_��A�%*O

prediction_loss)\?

reg_lossRHw<


total_lossJ9?


accuracy_1�G�>�36k]       a[��	��,_��A�%*O

prediction_loss   ?

reg_lossFHw<


total_loss!�?


accuracy_1   ?o�5�]       a[��	g
�,_��A�%*O

prediction_loss���>

reg_loss=Hw<


total_loss��>


accuracy_1��?��m]       a[��	�'�,_��A�%*O

prediction_loss=
�>

reg_loss3Hw<


total_loss��>


accuracy_1�z?��`2]       a[��	�@�,_��A�%*O

prediction_loss��>

reg_loss)Hw<


total_loss`?�>


accuracy_1q=
?�Ó�]       a[��	.X�,_��A�%*O

prediction_loss��>

reg_lossHw<


total_loss`?�>


accuracy_1q=
?Ok�Z]       a[��	�n�,_��A�%*O

prediction_loss�?

reg_lossHw<


total_loss��?


accuracy_1���>bǒ�]       a[��	3��,_��A�%*O

prediction_lossq=
?

reg_loss
Hw<


total_loss�?


accuracy_1��>�v^]       a[��	���,_��A�%*O

prediction_loss��>

reg_loss�Gw<


total_loss_?�>


accuracy_1q=
?]��]       a[��	Z��,_��A�%*O

prediction_loss=
�>

reg_loss�Gw<


total_loss}��>


accuracy_1�z?���]       a[��	�ׂ,_��A�%*O

prediction_loss)\?

reg_loss�Gw<


total_lossI9?


accuracy_1�G�>L�]       a[��	��,_��A�%*O

prediction_loss�z?

reg_loss�Gw<


total_lossX?


accuracy_1=
�>��%B]       a[��	��,_��A�%*O

prediction_loss   ?

reg_loss�Gw<


total_loss�?


accuracy_1   ?R��j]       a[��	Q2�,_��A�%*O

prediction_loss)\?

reg_loss�Gw<


total_lossH9?


accuracy_1�G�>�]       a[��	`[�,_��A�%*O

prediction_loss���>

reg_loss�Gw<


total_loss�|�>


accuracy_1�?��9]       a[��	-{�,_��A�%*O

prediction_loss�G�>

reg_loss�Gw<


total_loss��>


accuracy_1)\?͸��]       a[��	��,_��A�%*O

prediction_loss�G�>

reg_loss�Gw<


total_loss��>


accuracy_1)\?�F��]       a[��	��,_��A�%*O

prediction_lossR�?

reg_loss�Gw<


total_lossq�"?


accuracy_1\��>�8�0]       a[��	���,_��A�%*O

prediction_loss�z?

reg_loss�Gw<


total_loss�W?


accuracy_1=
�>*���]       a[��	Aׄ,_��A�%*O

prediction_loss��>

reg_loss�Gw<


total_loss[?�>


accuracy_1q=
?S��]       a[��	�X�,_��A�%*O

prediction_loss�?

reg_loss�Gw<


total_loss��?


accuracy_1���>Z��]       a[��	R��,_��A�%*O

prediction_loss��>

reg_lossyGw<


total_loss[?�>


accuracy_1q=
?�r_]       a[��	���,_��A�%*O

prediction_loss��?

reg_lossnGw<


total_loss�v?


accuracy_1���>�:��]       a[��	���,_��A�%*O

prediction_loss��>

reg_losseGw<


total_lossZ?�>


accuracy_1q=
?��.�]       a[��	F(�,_��A�%*O

prediction_loss��?

reg_lossZGw<


total_loss�v?


accuracy_1���>�e3�]       a[��	�P�,_��A�%*O

prediction_lossq=
?

reg_lossPGw<


total_loss�?


accuracy_1��>�N]       a[��	�y�,_��A�%*O

prediction_loss��?

reg_lossEGw<


total_loss�v?


accuracy_1���>I�Ao]       a[��	6��,_��A�%*O

prediction_lossq=
?

reg_loss;Gw<


total_loss�?


accuracy_1��>�oƧ]       a[��	��,_��A�%*O

prediction_loss��>

reg_loss1Gw<


total_lossY?�>


accuracy_1q=
?����]       a[��	@�,_��A�%*O

prediction_loss=
�>

reg_loss'Gw<


total_lossv��>


accuracy_1�z?_|�h]       a[��	�S�,_��A�%*O

prediction_loss�?

reg_lossGw<


total_loss��?


accuracy_1���>��I]       a[��	F|�,_��A�%*O

prediction_loss   ?

reg_lossGw<


total_loss�?


accuracy_1   ?�l}]       a[��	&��,_��A�%*O

prediction_loss=
�>

reg_lossGw<


total_lossu��>


accuracy_1�z?'Q�K]       a[��	��,_��A�%*O

prediction_loss�?

reg_loss�Fw<


total_loss��?


accuracy_1���>5��]       a[��	.U�,_��A�%*O

prediction_loss��(?

reg_loss�Fw<


total_loss��,?


accuracy_1{�>����]       a[��	e��,_��A�%*O

prediction_loss��?

reg_loss�Fw<


total_loss�v?


accuracy_1���>�R]       a[��	b��,_��A�%*O

prediction_loss��>

reg_loss�Fw<


total_lossV?�>


accuracy_1q=
?we	�]       a[��	���,_��A�%*O

prediction_lossq=
?

reg_loss�Fw<


total_loss�?


accuracy_1��>����]       a[��	y;�,_��A�%*O

prediction_loss���>

reg_loss�Fw<


total_loss�|�>


accuracy_1�?X�7]       a[��	l{�,_��A�%*O

prediction_loss\��>

reg_loss�Fw<


total_loss�I�>


accuracy_1R�?�}d�]       a[��	ޫ�,_��A�%*O

prediction_lossq=
?

reg_loss�Fw<


total_loss�?


accuracy_1��>g���]       a[��	/݉,_��A�%*O

prediction_loss���>

reg_loss�Fw<


total_loss�|�>


accuracy_1�?:�!�]       a[��	S�,_��A�%*O

prediction_lossq=
?

reg_loss�Fw<


total_loss�?


accuracy_1��>Z)>]       a[��	{.�,_��A�%*O

prediction_loss=
�>

reg_loss�Fw<


total_lossr��>


accuracy_1�z?��%]       a[��	R�,_��A�%*O

prediction_loss=
�>

reg_loss�Fw<


total_lossq��>


accuracy_1�z?�+�]       a[��	1}�,_��A�%*O

prediction_lossq=
?

reg_loss�Fw<


total_loss�?


accuracy_1��>�� e]       a[��	���,_��A�%*O

prediction_loss��>

reg_loss{Fw<


total_lossS?�>


accuracy_1q=
?���#]       a[��	K�,_��A�%*O

prediction_loss)\?

reg_lossoFw<


total_lossC9?


accuracy_1�G�>iKd�]       a[��	b.�,_��A�%*O

prediction_loss��>

reg_lossfFw<


total_lossR?�>


accuracy_1q=
?�]       a[��	�P�,_��A�%*O

prediction_loss�?

reg_loss[Fw<


total_loss��?


accuracy_1���>���]       a[��	�n�,_��A�%*O

prediction_loss�z?

reg_lossRFw<


total_loss�W?


accuracy_1=
�>��7:]       a[��	P��,_��A�%*O

prediction_loss�?

reg_lossEFw<


total_loss��?


accuracy_1���>f��]       a[��		��,_��A�%*O

prediction_loss\��>

reg_loss<Fw<


total_loss�I�>


accuracy_1R�?�� q]       a[��	�ԋ,_��A�%*O

prediction_lossq=
?

reg_loss1Fw<


total_loss�?


accuracy_1��>W(�$]       a[��	 (�,_��A�%*O

prediction_loss=
�>

reg_loss&Fw<


total_lossn��>


accuracy_1�z?%�7�]       a[��	h�,_��A�%*O

prediction_loss��?

reg_lossFw<


total_loss�v?


accuracy_1���>{�8]       a[��	9��,_��A�%*O

prediction_loss���>

reg_lossFw<


total_loss�|�>


accuracy_1�?�IfA]       a[��	���,_��A�%*O

prediction_loss��>

reg_lossFw<


total_lossO?�>


accuracy_1q=
?��p]       a[��	U݌,_��A�%*O

prediction_loss���>

reg_loss�Ew<


total_loss�|�>


accuracy_1�?��g�]       a[��	��,_��A�%*O

prediction_loss   ?

reg_loss�Ew<


total_loss�?


accuracy_1   ?��-�]       a[��	;�,_��A�%*O

prediction_loss�G�>

reg_loss�Ew<


total_loss��>


accuracy_1)\?�;�]       a[��	1[�,_��A�&*O

prediction_loss��(?

reg_loss�Ew<


total_loss��,?


accuracy_1{�>:�."]       a[��	w�,_��A�&*O

prediction_loss���>

reg_loss�Ew<


total_loss�|�>


accuracy_1�?��A]       a[��	���,_��A�&*O

prediction_loss�z?

reg_loss�Ew<


total_loss�W?


accuracy_1=
�>��]       a[��	���,_��A�&*O

prediction_lossq=
?

reg_loss�Ew<


total_loss�?


accuracy_1��>~G1�]       a[��	Ѝ,_��A�&*O

prediction_loss��?

reg_loss�Ew<


total_loss�v?


accuracy_1���>8�O]       a[��	z �,_��A�&*O

prediction_loss��>

reg_loss�Ew<


total_lossL?�>


accuracy_1q=
?�2�]       a[��	��,_��A�&*O

prediction_loss���>

reg_loss�Ew<


total_loss�|�>


accuracy_1�?�OF]       a[��	�:�,_��A�&*O

prediction_loss�G�>

reg_loss�Ew<


total_loss��>


accuracy_1)\?c8�]       a[��	OX�,_��A�&*O

prediction_loss)\?

reg_loss�Ew<


total_loss?9?


accuracy_1�G�>wβ]       a[��	�t�,_��A�&*O

prediction_loss�?

reg_loss�Ew<


total_loss��?


accuracy_1���>�/�O]       a[��	��,_��A�&*O

prediction_loss�?

reg_losstEw<


total_loss��?


accuracy_1���>��]       a[��	>��,_��A�&*O

prediction_loss�?

reg_losskEw<


total_loss��?


accuracy_1���>j-J�]       a[��	�Ŏ,_��A�&*O

prediction_loss�G�>

reg_loss`Ew<


total_loss��>


accuracy_1)\?Z-��]       a[��	vݎ,_��A�&*O

prediction_lossR�?

reg_lossUEw<


total_lossg�"?


accuracy_1\��>�!;]       a[��	�,_��A�&*O

prediction_loss��?

reg_lossLEw<


total_loss�v?


accuracy_1���>p^��]       a[��	�+�,_��A�&*O

prediction_loss)\?

reg_lossBEw<


total_loss>9?


accuracy_1�G�>�}�]       a[��	4G�,_��A�&*O

prediction_loss)\?

reg_loss8Ew<


total_loss>9?


accuracy_1�G�>3�.�]       a[��	�]�,_��A�&*O

prediction_loss\��>

reg_loss/Ew<


total_loss�I�>


accuracy_1R�??hЏ]       a[��	�x�,_��A�&*O

prediction_loss   ?

reg_loss#Ew<


total_loss�?


accuracy_1   ?�1:]       a[��	S��,_��A�&*O

prediction_loss   ?

reg_lossEw<


total_loss�?


accuracy_1   ?���d]       a[��	ٳ�,_��A�&*O

prediction_lossR�?

reg_lossEw<


total_lossf�"?


accuracy_1\��>�@��]       a[��	ˏ,_��A�&*O

prediction_loss�G�>

reg_lossEw<


total_loss��>


accuracy_1)\?��K]       a[��	�,_��A�&*O

prediction_loss���>

reg_loss�Dw<


total_loss�|�>


accuracy_1�?.g�]       a[��	�
�,_��A�&*O

prediction_lossR�?

reg_loss�Dw<


total_lossf�"?


accuracy_1\��>�_]       a[��	�1�,_��A�&*O

prediction_loss�Q�>

reg_loss�Dw<


total_loss�>


accuracy_1
�#?�V$�]       a[��	W�,_��A�&*O

prediction_loss�z?

reg_loss�Dw<


total_loss�W?


accuracy_1=
�>�$�]       a[��	�q�,_��A�&*O

prediction_loss   ?

reg_loss�Dw<


total_loss�?


accuracy_1   ?�#�K]       a[��	i��,_��A�&*O

prediction_loss)\?

reg_loss�Dw<


total_loss<9?


accuracy_1�G�>�i��]       a[��	���,_��A�&*O

prediction_loss�G�>

reg_loss�Dw<


total_loss��>


accuracy_1)\?�6S�]       a[��	��,_��A�&*O

prediction_loss���>

reg_loss�Dw<


total_loss�|�>


accuracy_1�?6ኃ]       a[��	�ؐ,_��A�&*O

prediction_loss)\?

reg_loss�Dw<


total_loss<9?


accuracy_1�G�>*� �]       a[��	<��,_��A�&*O

prediction_loss=
�>

reg_loss�Dw<


total_lossb��>


accuracy_1�z?�[�]       a[��	.�,_��A�&*O

prediction_loss�G�>

reg_loss�Dw<


total_loss��>


accuracy_1)\?�X��]       a[��	r1�,_��A�&*O

prediction_lossq=
?

reg_loss�Dw<


total_loss�?


accuracy_1��>]��]       a[��	V�,_��A�&*O

prediction_lossq=
?

reg_lossDw<


total_loss�?


accuracy_1��>�K�]       a[��	�n�,_��A�&*O

prediction_loss�G�>

reg_losstDw<


total_loss��>


accuracy_1)\?���]       a[��	"��,_��A�&*O

prediction_loss���>

reg_lossjDw<


total_loss�|�>


accuracy_1�?��b]       a[��	��,_��A�&*O

prediction_lossq=
?

reg_lossVDw<


total_loss�?


accuracy_1��>ϸ�x]       a[��	���,_��A�&*O

prediction_loss���>

reg_lossKDw<


total_loss�|�>


accuracy_1�?��c�]       a[��	'�,_��A�&*O

prediction_loss���>

reg_lossADw<


total_loss�|�>


accuracy_1�?0�_]       a[��	�5�,_��A�&*O

prediction_loss�G�>

reg_loss6Dw<


total_loss��>


accuracy_1)\?bc�]       a[��	c`�,_��A�&*O

prediction_loss=
�>

reg_loss-Dw<


total_loss^��>


accuracy_1�z?wn>�]       a[��	V~�,_��A�&*O

prediction_lossq=
?

reg_loss$Dw<


total_loss�?


accuracy_1��>v���]       a[��	ᘒ,_��A�&*O

prediction_loss��?

reg_lossDw<


total_loss�v?


accuracy_1���>����]       a[��	1��,_��A�&*O

prediction_loss�G�>

reg_lossDw<


total_loss��>


accuracy_1)\?��$]       a[��	̒,_��A�&*O

prediction_loss���>

reg_lossDw<


total_loss�|�>


accuracy_1�?�R��]       a[��	��,_��A�&*O

prediction_loss���>

reg_loss�Cw<


total_loss�|�>


accuracy_1�?&*]       a[��	���,_��A�&*O

prediction_loss���>

reg_loss�Cw<


total_loss�|�>


accuracy_1�?��]       a[��	b�,_��A�&*O

prediction_loss��>

reg_loss�Cw<


total_loss>?�>


accuracy_1q=
?��]       a[��	H7�,_��A�&*O

prediction_loss�?

reg_loss�Cw<


total_loss��?


accuracy_1���>ws�]       a[��	dY�,_��A�&*O

prediction_loss{.?

reg_loss�Cw<


total_loss��1?


accuracy_1
ף>9i�]       a[��	~�,_��A�&*O

prediction_loss\��>

reg_loss�Cw<


total_losszI�>


accuracy_1R�?d��]       a[��	���,_��A�&*O

prediction_loss�?

reg_loss�Cw<


total_loss��?


accuracy_1���> ��]       a[��	���,_��A�&*O

prediction_lossR�?

reg_loss�Cw<


total_lossa�"?


accuracy_1\��>��e�]       a[��	��,_��A�&*O

prediction_loss��>

reg_loss�Cw<


total_loss<?�>


accuracy_1q=
?Lp�]       a[��	��,_��A�&*O

prediction_loss�z?

reg_loss�Cw<


total_loss�W?


accuracy_1=
�>r7��]       a[��	|B�,_��A�&*O

prediction_loss   ?

reg_loss�Cw<


total_loss�?


accuracy_1   ?�q"�]       a[��	�o�,_��A�&*O

prediction_loss���>

reg_loss�Cw<


total_loss��>


accuracy_1��?U�E�]       a[��	Y��,_��A�&*O

prediction_loss�G�>

reg_loss�Cw<


total_loss��>


accuracy_1)\?[Љ�]       a[��	��,_��A�&*O

prediction_loss���>

reg_lossuCw<


total_loss�|�>


accuracy_1�?V���]       a[��	k�,_��A�&*O

prediction_loss   ?

reg_losslCw<


total_loss�?


accuracy_1   ?��A�]       a[��	�-�,_��A�&*O

prediction_loss���>

reg_loss`Cw<


total_loss�|�>


accuracy_1�? �B�]       a[��	.V�,_��A�&*O

prediction_loss���>

reg_lossWCw<


total_loss�|�>


accuracy_1�?`8'�]       a[��	�~�,_��A�&*O

prediction_loss��>

reg_lossKCw<


total_loss9?�>


accuracy_1q=
?
Ԥ�]       a[��	���,_��A�&*O

prediction_loss   ?

reg_lossACw<


total_loss�?


accuracy_1   ?F��Z]       a[��	�ŕ,_��A�&*O

prediction_loss���>

reg_loss6Cw<


total_loss�|�>


accuracy_1�?x��]       a[��	���,_��A�&*O

prediction_loss�?

reg_loss,Cw<


total_loss��?


accuracy_1���>&��]       a[��	;��,_��A�&*O

prediction_loss�?

reg_loss!Cw<


total_loss��?


accuracy_1���>`�+]       a[��	~�,_��A�&*O

prediction_loss��>

reg_lossCw<


total_loss8?�>


accuracy_1q=
?�L�4]       a[��	:�,_��A�&*O

prediction_loss\��>

reg_lossCw<


total_losstI�>


accuracy_1R�?�y.�]       a[��	�S�,_��A�&*O

prediction_loss���>

reg_lossCw<


total_loss�|�>


accuracy_1�?��']       a[��	�u�,_��A�&*O

prediction_loss���>

reg_loss�Bw<


total_loss��>


accuracy_1��?��]       a[��	���,_��A�&*O

prediction_loss��>

reg_loss�Bw<


total_loss7?�>


accuracy_1q=
?r���]       a[��	n��,_��A�&*O

prediction_loss
ף>

reg_loss�Bw<


total_loss!��>


accuracy_1{.?�c`]       a[��	jܖ,_��A�&*O

prediction_loss�z?

reg_loss�Bw<


total_loss�W?


accuracy_1=
�>oM�d]       a[��	���,_��A�&*O

prediction_loss=
�>

reg_loss�Bw<


total_lossT��>


accuracy_1�z?���]       a[��	T�,_��A�&*O

prediction_loss�?

reg_loss�Bw<


total_loss��?


accuracy_1���>�Hh]       a[��	6�,_��A�&*O

prediction_loss�?

reg_loss�Bw<


total_loss��?


accuracy_1���>�"?*]       a[��	@O�,_��A�&*O

prediction_loss   ?

reg_loss�Bw<


total_loss�?


accuracy_1   ?Ts��]       a[��	m�,_��A�&*O

prediction_loss��>

reg_loss�Bw<


total_loss4?�>


accuracy_1q=
?��y]       a[��	���,_��A�&*O

prediction_loss)\?

reg_loss�Bw<


total_loss39?


accuracy_1�G�>7�x+]       a[��	и�,_��A�&*O

prediction_loss��>

reg_loss�Bw<


total_loss4?�>


accuracy_1q=
?�S@J]       a[��	ԗ,_��A�&*O

prediction_loss�G�>

reg_loss�Bw<


total_loss��>


accuracy_1)\?&�%]       a[��	p�,_��A�&*O

prediction_loss��>

reg_loss~Bw<


total_loss3?�>


accuracy_1q=
?�Ya�]       a[��	x	�,_��A�&*O

prediction_loss���>

reg_losstBw<


total_loss�|�>


accuracy_1�?R��]       a[��	�(�,_��A�&*O

prediction_loss\��>

reg_lossgBw<


total_lossoI�>


accuracy_1R�?�/��]       a[��	�H�,_��A�&*O

prediction_loss   ?

reg_loss_Bw<


total_loss	�?


accuracy_1   ?��\�]       a[��	c�,_��A�&*O

prediction_loss)\?

reg_lossVBw<


total_loss29?


accuracy_1�G�>�μ�]       a[��	��,_��A�&*O

prediction_loss��>

reg_lossJBw<


total_loss1?�>


accuracy_1q=
?��]       a[��	x��,_��A�&*O

prediction_lossq=
?

reg_lossBBw<


total_lossz?


accuracy_1��>~�5�]       a[��	�֘,_��A�&*O

prediction_loss�z?

reg_loss6Bw<


total_loss�W?


accuracy_1=
�>b%n]       a[��	=�,_��A�&*O

prediction_loss��>

reg_loss,Bw<


total_loss0?�>


accuracy_1q=
?G��]       a[��	
�,_��A�&*O

prediction_loss���>

reg_loss!Bw<


total_loss�|�>


accuracy_1�?�|Uv]       a[��	�%�,_��A�&*O

prediction_lossq=
?

reg_lossBw<


total_lossy?


accuracy_1��>���]       a[��	�E�,_��A�&*O

prediction_loss���>

reg_lossBw<


total_loss�|�>


accuracy_1�?o�V]       a[��	�^�,_��A�&*O

prediction_loss���>

reg_lossBw<


total_loss�|�>


accuracy_1�?��t�]       a[��	��,_��A�&*O

prediction_loss��>

reg_loss�Aw<


total_loss/?�>


accuracy_1q=
?�=�]       a[��	
��,_��A�&*O

prediction_loss��?

reg_loss�Aw<


total_loss�v?


accuracy_1���>@�6�]       a[��	(љ,_��A�&*O

prediction_loss��>

reg_loss�Aw<


total_loss.?�>


accuracy_1q=
?�٣+]       a[��	���,_��A�&*O

prediction_loss   ?

reg_loss�Aw<


total_loss�?


accuracy_1   ?�2&�]       a[��	��,_��A�&*O

prediction_loss�z?

reg_loss�Aw<


total_loss�W?


accuracy_1=
�>"��]       a[��	D3�,_��A�&*O

prediction_loss   ?

reg_loss�Aw<


total_loss�?


accuracy_1   ?�Q�]       a[��	3Q�,_��A�&*O

prediction_loss)\?

reg_loss�Aw<


total_loss09?


accuracy_1�G�>��U]       a[��	{j�,_��A�&*O

prediction_loss�G�>

reg_loss�Aw<


total_loss��>


accuracy_1)\?B3&]       a[��	@��,_��A�&*O

prediction_loss��>

reg_loss�Aw<


total_loss,?�>


accuracy_1q=
?���]       a[��	ʦ�,_��A�&*O

prediction_loss   ?

reg_loss�Aw<


total_loss�?


accuracy_1   ?B[�]       a[��	kҚ,_��A�&*O

prediction_loss��?

reg_loss�Aw<


total_loss�v?


accuracy_1���>�G;�]       a[��	M�,_��A�&*O

prediction_loss���>

reg_loss�Aw<


total_loss�|�>


accuracy_1�?�)�]       a[��	��,_��A�&*O

prediction_loss��?

reg_losszAw<


total_loss�v?


accuracy_1���>q�9�]       a[��	�)�,_��A�&*O

prediction_loss�?

reg_lossoAw<


total_loss��?


accuracy_1���>���]       a[��	oJ�,_��A�&*O

prediction_loss   ?

reg_losseAw<


total_loss�?


accuracy_1   ?�r��]       a[��	�i�,_��A�&*O

prediction_loss�?

reg_loss[Aw<


total_loss��?


accuracy_1���>�:�]       a[��	s��,_��A�&*O

prediction_loss)\?

reg_lossPAw<


total_loss.9?


accuracy_1�G�>=?�H]       a[��	ԝ�,_��A�&*O

prediction_loss��>

reg_lossGAw<


total_loss)?�>


accuracy_1q=
?��o�]       a[��	���,_��A�&*O

prediction_loss��?

reg_loss<Aw<


total_loss�v?


accuracy_1���>f�'�]       a[��	���,_��A�&*O

prediction_loss���>

reg_loss2Aw<


total_loss�|�>


accuracy_1�?�u�]       a[��	~�,_��A�&*O

prediction_loss=
�>

reg_loss'Aw<


total_lossF��>


accuracy_1�z?r��]       a[��	D�,_��A�&*O

prediction_loss
ף>

reg_lossAw<


total_loss��>


accuracy_1{.?�$D>]       a[��	�d�,_��A�&*O

prediction_loss)\?

reg_lossAw<


total_loss-9?


accuracy_1�G�>J~G�]       a[��	���,_��A�&*O

prediction_loss��>

reg_loss	Aw<


total_loss'?�>


accuracy_1q=
?����]       a[��	Ӥ�,_��A�&*O

prediction_loss�?

reg_loss�@w<


total_loss��?


accuracy_1���>	e�y]       a[��	�,_��A�&*O

prediction_loss�G�>

reg_loss�@w<


total_loss��>


accuracy_1)\?��D�]       a[��	��,_��A�&*O

prediction_loss)\?

reg_loss�@w<


total_loss-9?


accuracy_1�G�>"T�]       a[��	[	�,_��A�&*O

prediction_loss   ?

reg_loss�@w<


total_loss�?


accuracy_1   ?��`]       a[��	�%�,_��A�&*O

prediction_loss�?

reg_loss�@w<


total_loss��?


accuracy_1���>J�'Q]       a[��	)A�,_��A�&*O

prediction_loss���>

reg_loss�@w<


total_loss�|�>


accuracy_1�?�~�]       a[��	p_�,_��A�&*O

prediction_lossq=
?

reg_loss�@w<


total_losst?


accuracy_1��>�J?�]       a[��	�~�,_��A�&*O

prediction_loss   ?

reg_loss�@w<


total_loss�?


accuracy_1   ?�CU]       a[��	|��,_��A�'*O

prediction_loss��?

reg_loss�@w<


total_loss�v?


accuracy_1���>ͨ7]       a[��	��,_��A�'*O

prediction_loss���>

reg_loss�@w<


total_loss�|�>


accuracy_1�?��]       a[��	O�,_��A�'*O

prediction_loss��>

reg_loss�@w<


total_loss$?�>


accuracy_1q=
?r��]       a[��	6#�,_��A�'*O

prediction_loss   ?

reg_loss�@w<


total_loss�?


accuracy_1   ?pPc�]       a[��	�H�,_��A�'*O

prediction_loss���>

reg_loss�@w<


total_loss�|�>


accuracy_1�?�KQ�]       a[��	jj�,_��A�'*O

prediction_loss���>

reg_lossz@w<


total_loss�|�>


accuracy_1�?@��]       a[��	���,_��A�'*O

prediction_lossq=
?

reg_lossq@w<


total_losss?


accuracy_1��>w1�]       a[��	m��,_��A�'*O

prediction_lossq=
?

reg_lossg@w<


total_losss?


accuracy_1��>l9E�]       a[��	)О,_��A�'*O

prediction_loss���>

reg_loss]@w<


total_lossІ�>


accuracy_1��?�]       a[��	l?�,_��A�'*O

prediction_loss���>

reg_lossG@w<


total_loss�|�>


accuracy_1�?a�]       a[��	g_�,_��A�'*O

prediction_loss)\?

reg_loss>@w<


total_loss*9?


accuracy_1�G�>�$�]       a[��	�,_��A�'*O

prediction_loss�z?

reg_loss3@w<


total_loss�W?


accuracy_1=
�>F�v]       a[��	���,_��A�'*O

prediction_loss   ?

reg_loss)@w<


total_loss�?


accuracy_1   ?c)�s]       a[��	��,_��A�'*O

prediction_loss��>

reg_loss @w<


total_loss ?�>


accuracy_1q=
?��6]       a[��	�֟,_��A�'*O

prediction_loss   ?

reg_loss@w<


total_loss �?


accuracy_1   ?3��]       a[��	���,_��A�'*O

prediction_loss�z?

reg_loss
@w<


total_loss�W?


accuracy_1=
�>�OVV]       a[��	��,_��A�'*O

prediction_loss   ?

reg_loss@w<


total_loss �?


accuracy_1   ?����]       a[��	�9�,_��A�'*O

prediction_loss=
�>

reg_loss�?w<


total_loss=��>


accuracy_1�z?dr�q]       a[��	�X�,_��A�'*O

prediction_loss���>

reg_loss�?w<


total_loss�|�>


accuracy_1�?�
]       a[��	�v�,_��A�'*O

prediction_loss�?

reg_loss�?w<


total_loss��?


accuracy_1���>�?s]       a[��	?��,_��A�'*O

prediction_lossq=
?

reg_loss�?w<


total_lossp?


accuracy_1��>q+�]       a[��	���,_��A�'*O

prediction_loss�?

reg_loss�?w<


total_loss��?


accuracy_1���>i��J]       a[��	:̠,_��A�'*O

prediction_loss)\?

reg_loss�?w<


total_loss(9?


accuracy_1�G�>�[W�]       a[��	��,_��A�'*O

prediction_loss��?

reg_loss�?w<


total_loss�v?


accuracy_1���>�b�L]       a[��	1�,_��A�'*O

prediction_loss���>

reg_loss�?w<


total_loss�S�>


accuracy_1333?��]       a[��	�Q�,_��A�'*O

prediction_loss)\?

reg_loss�?w<


total_loss(9?


accuracy_1�G�>�g��]       a[��	v�,_��A�'*O

prediction_loss   ?

reg_loss�?w<


total_loss��?


accuracy_1   ?�%]       a[��	���,_��A�'*O

prediction_loss��>

reg_loss�?w<


total_loss?�>


accuracy_1q=
?fR�_]       a[��	���,_��A�'*O

prediction_loss
ף>

reg_loss�?w<


total_loss��>


accuracy_1{.?T��]       a[��	��,_��A�'*O

prediction_loss���>

reg_lossy?w<


total_lossɆ�>


accuracy_1��?t~�]       a[��	�4�,_��A�'*O

prediction_loss)\?

reg_losso?w<


total_loss'9?


accuracy_1�G�>�_#]       a[��	f�,_��A�'*O

prediction_loss)\?

reg_losse?w<


total_loss'9?


accuracy_1�G�>&��]       a[��	���,_��A�'*O

prediction_loss�?

reg_loss\?w<


total_loss��?


accuracy_1���>��U]       a[��	g��,_��A�'*O

prediction_loss
�#?

reg_lossQ?w<


total_loss�'?


accuracy_1�Q�>�p��]       a[��	˽�,_��A�'*O

prediction_lossq=
?

reg_lossF?w<


total_lossn?


accuracy_1��>���o]       a[��	4ڢ,_��A�'*O

prediction_loss���>

reg_loss=?w<


total_loss�|�>


accuracy_1�?��#]       a[��	���,_��A�'*O

prediction_loss�?

reg_loss2?w<


total_loss��?


accuracy_1���>�=�K]       a[��	��,_��A�'*O

prediction_loss{�>

reg_loss(?w<


total_losstε>


accuracy_1��(?vbE]       a[��	28�,_��A�'*O

prediction_loss)\?

reg_loss?w<


total_loss%9?


accuracy_1�G�>�NJ]       a[��	Y�,_��A�'*O

prediction_loss��>

reg_loss?w<


total_loss?�>


accuracy_1q=
?��r�]       a[��	�u�,_��A�'*O

prediction_loss���>

reg_loss
?w<


total_lossņ�>


accuracy_1��?4�]       a[��	x��,_��A�'*O

prediction_loss)\?

reg_loss ?w<


total_loss%9?


accuracy_1�G�>ͅ�]       a[��	곣,_��A�'*O

prediction_loss   ?

reg_loss�>w<


total_loss��?


accuracy_1   ?��43]       a[��	�ѣ,_��A�'*O

prediction_loss�?

reg_loss�>w<


total_loss��?


accuracy_1���>����]       a[��	t��,_��A�'*O

prediction_loss)\?

reg_loss�>w<


total_loss$9?


accuracy_1�G�>�-��]       a[��	�	�,_��A�'*O

prediction_loss)\?

reg_loss�>w<


total_loss$9?


accuracy_1�G�>�`��]       a[��	J'�,_��A�'*O

prediction_loss�z?

reg_loss�>w<


total_loss�W?


accuracy_1=
�>篟�]       a[��	�O�,_��A�'*O

prediction_loss�z?

reg_loss�>w<


total_loss�W?


accuracy_1=
�>[I�^]       a[��	p�,_��A�'*O

prediction_loss��?

reg_loss�>w<


total_loss�v?


accuracy_1���>�5R{]       a[��	���,_��A�'*O

prediction_loss���>

reg_loss�>w<


total_loss�>


accuracy_1��?�a(]       a[��	���,_��A�'*O

prediction_loss��>

reg_loss�>w<


total_loss?�>


accuracy_1q=
?�m��]       a[��	HĤ,_��A�'*O

prediction_lossR�?

reg_loss�>w<


total_lossL�"?


accuracy_1\��>#uw�]       a[��	aݤ,_��A�'*O

prediction_loss   ?

reg_loss�>w<


total_loss��?


accuracy_1   ?b~1]       a[��	a��,_��A�'*O

prediction_loss)\?

reg_loss�>w<


total_loss#9?


accuracy_1�G�>4o��]       a[��	��,_��A�'*O

prediction_loss���>

reg_lossx>w<


total_loss�|�>


accuracy_1�?�7O�]       a[��	�4�,_��A�'*O

prediction_loss��>

reg_losso>w<


total_loss?�>


accuracy_1q=
?����]       a[��	�V�,_��A�'*O

prediction_loss��?

reg_lossd>w<


total_loss�v?


accuracy_1���>U��]       a[��	�z�,_��A�'*O

prediction_lossq=
?

reg_lossZ>w<


total_lossj?


accuracy_1��>G��B]       a[��	,��,_��A�'*O

prediction_loss�?

reg_lossQ>w<


total_loss��?


accuracy_1���>�@t]       a[��	���,_��A�'*O

prediction_loss)\?

reg_lossG>w<


total_loss"9?


accuracy_1�G�>R�]       a[��	}Υ,_��A�'*O

prediction_loss�G�>

reg_loss;>w<


total_loss��>


accuracy_1)\?>�I(]       a[��	��,_��A�'*O

prediction_loss)\?

reg_loss2>w<


total_loss"9?


accuracy_1�G�>l�&�]       a[��	~��,_��A�'*O

prediction_loss�z?

reg_loss&>w<


total_loss�W?


accuracy_1=
�>f�`]       a[��	��,_��A�'*O

prediction_lossq=
?

reg_loss>w<


total_lossi?


accuracy_1��>7���]       a[��	1�,_��A�'*O

prediction_loss��>

reg_loss>w<


total_loss?�>


accuracy_1q=
?�	�]       a[��	�I�,_��A�'*O

prediction_loss���>

reg_loss>w<


total_loss|�>


accuracy_1�?マ�]       a[��	�j�,_��A�'*O

prediction_loss�?

reg_loss�=w<


total_loss��?


accuracy_1���><�d]       a[��	���,_��A�'*O

prediction_loss   ?

reg_loss�=w<


total_loss��?


accuracy_1   ?dX�]       a[��	ʣ�,_��A�'*O

prediction_loss��>

reg_loss�=w<


total_loss?�>


accuracy_1q=
?����]       a[��	8��,_��A�'*O

prediction_loss�G�>

reg_loss�=w<


total_loss��>


accuracy_1)\?�*2]       a[��	pѦ,_��A�'*O

prediction_loss�z?

reg_loss�=w<


total_loss�W?


accuracy_1=
�>�_7�]       a[��	��,_��A�'*O

prediction_loss�G�>

reg_loss�=w<


total_loss��>


accuracy_1)\?Q�|]       a[��	h�,_��A�'*O

prediction_loss   ?

reg_loss�=w<


total_loss��?


accuracy_1   ?��ol]       a[��	�$�,_��A�'*O

prediction_loss�G�>

reg_loss�=w<


total_loss��>


accuracy_1)\?β
�]       a[��	B?�,_��A�'*O

prediction_lossR�?

reg_loss�=w<


total_lossI�"?


accuracy_1\��>`G`�]       a[��	�W�,_��A�'*O

prediction_loss��?

reg_loss�=w<


total_loss�v?


accuracy_1���>���]       a[��	q�,_��A�'*O

prediction_loss�G�>

reg_loss�=w<


total_loss��>


accuracy_1)\?rA��]       a[��	O��,_��A�'*O

prediction_loss)\?

reg_loss�=w<


total_loss9?


accuracy_1�G�>�gL]       a[��	���,_��A�'*O

prediction_lossR�?

reg_loss�=w<


total_lossH�"?


accuracy_1\��>�a�~]       a[��	�ާ,_��A�'*O

prediction_lossq=
?

reg_lossy=w<


total_lossg?


accuracy_1��>���]       a[��	@��,_��A�'*O

prediction_lossR�?

reg_losso=w<


total_lossH�"?


accuracy_1\��>�K-�]       a[��	���,_��A�'*O

prediction_lossR�?

reg_lossd=w<


total_lossH�"?


accuracy_1\��>Z�]       a[��	�Z�,_��A�'*O

prediction_loss)\?

reg_lossZ=w<


total_loss9?


accuracy_1�G�>�Dl]       a[��	��,_��A�'*O

prediction_loss   ?

reg_lossO=w<


total_loss��?


accuracy_1   ?�K-.]       a[��	lz�,_��A�'*O

prediction_loss�z?

reg_lossD=w<


total_loss�W?


accuracy_1=
�>����]       a[��	d̪,_��A�'*O

prediction_loss�?

reg_loss:=w<


total_loss��?


accuracy_1���>K��]       a[��	n��,_��A�'*O

prediction_loss��>

reg_loss/=w<


total_loss?�>


accuracy_1q=
?��]       a[��	J'�,_��A�'*O

prediction_loss=
�>

reg_loss$=w<


total_loss&��>


accuracy_1�z?UӶ/]       a[��	�Y�,_��A�'*O

prediction_loss��>

reg_loss=w<


total_loss?�>


accuracy_1q=
?�B �]       a[��	=�,_��A�'*O

prediction_lossq=
?

reg_loss=w<


total_losse?


accuracy_1��>HfZ�]       a[��	"��,_��A�'*O

prediction_loss)\?

reg_loss=w<


total_loss9?


accuracy_1�G�>i�<]       a[��	�,_��A�'*O

prediction_loss��>

reg_loss�<w<


total_loss?�>


accuracy_1q=
?2�]       a[��	E�,_��A�'*O

prediction_loss)\?

reg_loss�<w<


total_loss9?


accuracy_1�G�>��|]       a[��	�@�,_��A�'*O

prediction_loss
�#?

reg_loss�<w<


total_loss��'?


accuracy_1�Q�>ߧ{�]       a[��	p�,_��A�'*O

prediction_loss���>

reg_loss�<w<


total_loss���>


accuracy_1��?, T�]       a[��	h��,_��A�'*O

prediction_loss   ?

reg_loss�<w<


total_loss��?


accuracy_1   ?� N]       a[��	�Ԭ,_��A�'*O

prediction_lossR�?

reg_loss�<w<


total_lossE�"?


accuracy_1\��>̅;R]       a[��	��,_��A�'*O

prediction_lossq=
?

reg_loss�<w<


total_lossd?


accuracy_1��>j�Ү]       a[��	t�,_��A�'*O

prediction_loss�?

reg_loss�<w<


total_loss��?


accuracy_1���>��۳]       a[��		��,_��A�'*O

prediction_loss=
�>

reg_loss�<w<


total_loss"��>


accuracy_1�z?� ��]       a[��	c�,_��A�'*O

prediction_loss�?

reg_loss�<w<


total_loss��?


accuracy_1���>���E]       a[��	��,_��A�'*O

prediction_loss   ?

reg_loss�<w<


total_loss��?


accuracy_1   ?��'�]       a[��	�
�,_��A�'*O

prediction_loss)\?

reg_loss�<w<


total_loss9?


accuracy_1�G�>/H]       a[��	�.�,_��A�'*O

prediction_loss�z?

reg_loss�<w<


total_loss�W?


accuracy_1=
�>t���]       a[��	���,_��A�'*O

prediction_loss�?

reg_lossw<w<


total_loss��?


accuracy_1���>�Z��]       a[��	E��,_��A�'*O

prediction_loss�Q�>

reg_lossn<w<


total_loss��>


accuracy_1
�#?kR��]       a[��	�0�,_��A�'*O

prediction_loss�?

reg_lossd<w<


total_loss��?


accuracy_1���>3��]       a[��	c_�,_��A�'*O

prediction_loss�?

reg_lossX<w<


total_loss��?


accuracy_1���>՞e]       a[��	��,_��A�'*O

prediction_loss�z?

reg_lossO<w<


total_loss�W?


accuracy_1=
�>����]       a[��	��,_��A�'*O

prediction_loss   ?

reg_loss:<w<


total_loss��?


accuracy_1   ?����]       a[��	�ȱ,_��A�'*O

prediction_loss���>

reg_loss/<w<


total_lossp|�>


accuracy_1�?`]       a[��	��,_��A�'*O

prediction_lossR�?

reg_loss%<w<


total_lossC�"?


accuracy_1\��>���]       a[��	� �,_��A�'*O

prediction_loss\��>

reg_loss<w<


total_loss=I�>


accuracy_1R�?��
4]       a[��	���,_��A�'*O

prediction_loss�?

reg_loss<w<


total_loss��?


accuracy_1���>H}|V]       a[��	Ҭ�,_��A�'*O

prediction_loss�?

reg_loss<w<


total_loss��?


accuracy_1���>��!]       a[��	��,_��A�'*O

prediction_loss�G�>

reg_loss�;w<


total_loss��>


accuracy_1)\?�V�B]       a[��	zP�,_��A�'*O

prediction_loss�?

reg_loss�;w<


total_loss��?


accuracy_1���>N�o]       a[��	��,_��A�'*O

prediction_lossq=
?

reg_loss�;w<


total_lossa?


accuracy_1��>1��]       a[��	�3�,_��A�'*O

prediction_loss���>

reg_loss�;w<


total_loss���>


accuracy_1��?H�ʞ]       a[��	�y�,_��A�'*O

prediction_loss���>

reg_loss�;w<


total_loss���>


accuracy_1��?����]       a[��	줴,_��A�'*O

prediction_lossq=
?

reg_loss�;w<


total_loss`?


accuracy_1��>��;]       a[��	NѴ,_��A�'*O

prediction_loss���>

reg_loss�;w<


total_lossm|�>


accuracy_1�?��4]       a[��	��,_��A�'*O

prediction_loss\��>

reg_loss�;w<


total_loss:I�>


accuracy_1R�?��]       a[��	��,_��A�'*O

prediction_loss���>

reg_loss�;w<


total_loss���>


accuracy_1��?�w�H]       a[��	�`�,_��A�'*O

prediction_lossR�?

reg_loss�;w<


total_loss@�"?


accuracy_1\��>
���]       a[��	-��,_��A�'*O

prediction_lossq=
?

reg_loss�;w<


total_loss_?


accuracy_1��>���T]       a[��	ϼ�,_��A�'*O

prediction_loss   ?

reg_loss�;w<


total_loss��?


accuracy_1   ?X}��]       a[��	]��,_��A�'*O

prediction_loss�z?

reg_loss};w<


total_loss�W?


accuracy_1=
�>�]]       a[��	tb�,_��A�(*O

prediction_loss�?

reg_lossr;w<


total_loss��?


accuracy_1���>�G&]       a[��	��,_��A�(*O

prediction_loss=
�>

reg_losse;w<


total_loss��>


accuracy_1�z?u�B�]       a[��	'��,_��A�(*O

prediction_loss=
�>

reg_loss\;w<


total_loss��>


accuracy_1�z?�Ĉ]       a[��	z�,_��A�(*O

prediction_loss���>

reg_lossR;w<


total_lossj|�>


accuracy_1�?g�D�]       a[��	��,_��A�(*O

prediction_loss333?

reg_lossH;w<


total_loss 7?


accuracy_1���> �<	]       a[��	�1�,_��A�(*O

prediction_loss���>

reg_loss<;w<


total_lossi|�>


accuracy_1�?����]       a[��	eķ,_��A�(*O

prediction_lossq=
?

reg_loss3;w<


total_loss^?


accuracy_1��>rǶ]       a[��	a�,_��A�(*O

prediction_loss=
�>

reg_loss';w<


total_loss��>


accuracy_1�z?{U$�]       a[��	��,_��A�(*O

prediction_loss�?

reg_loss;w<


total_loss��?


accuracy_1���>��{]       a[��	�@�,_��A�(*O

prediction_loss���>

reg_loss;w<


total_lossh|�>


accuracy_1�?��{�]       a[��	x��,_��A�(*O

prediction_loss)\?

reg_loss	;w<


total_loss9?


accuracy_1�G�>T�i�]       a[��	
��,_��A�(*O

prediction_loss��>

reg_loss ;w<


total_loss�>�>


accuracy_1q=
?�Gȍ]       a[��	�ڸ,_��A�(*O

prediction_lossq=
?

reg_loss�:w<


total_loss]?


accuracy_1��>cHHR]       a[��	Z�,_��A�(*O

prediction_lossR�?

reg_loss�:w<


total_loss>�"?


accuracy_1\��>�}W]       a[��	;5�,_��A�(*O

prediction_lossq=
?

reg_loss�:w<


total_loss\?


accuracy_1��>1��;]       a[��	�R�,_��A�(*O

prediction_loss��?

reg_loss�:w<


total_loss�v?


accuracy_1���>���`]       a[��	*��,_��A�(*O

prediction_loss���>

reg_loss�:w<


total_losse|�>


accuracy_1�?H�]       a[��	�ֹ,_��A�(*O

prediction_loss   ?

reg_loss�:w<


total_loss��?


accuracy_1   ?���]       a[��	��,_��A�(*O

prediction_loss�G�>

reg_loss�:w<


total_loss��>


accuracy_1)\?���]       a[��	�H�,_��A�(*O

prediction_lossq=
?

reg_loss�:w<


total_loss\?


accuracy_1��>���]       a[��	o��,_��A�(*O

prediction_loss   ?

reg_loss�:w<


total_loss��?


accuracy_1   ?
���]       a[��	�ƺ,_��A�(*O

prediction_loss   ?

reg_loss�:w<


total_loss��?


accuracy_1   ?��2�]       a[��	��,_��A�(*O

prediction_loss�?

reg_loss�:w<


total_loss��?


accuracy_1���>rK(]       a[��	�0�,_��A�(*O

prediction_loss���>

reg_loss�:w<


total_lossc|�>


accuracy_1�?�죁]       a[��	\Y�,_��A�(*O

prediction_loss�G�>

reg_loss{:w<


total_loss��>


accuracy_1)\?���]       a[��	�x�,_��A�(*O

prediction_loss���>

reg_lossp:w<


total_lossb|�>


accuracy_1�?�2�]       a[��	��,_��A�(*O

prediction_lossq=
?

reg_lossg:w<


total_loss[?


accuracy_1��>U�	;]       a[��	>ͻ,_��A�(*O

prediction_loss�?

reg_loss]:w<


total_loss��?


accuracy_1���>�M�]       a[��	L��,_��A�(*O

prediction_loss=
�>

reg_lossS:w<


total_loss��>


accuracy_1�z?�R U]       a[��	�H�,_��A�(*O

prediction_loss=
�>

reg_lossJ:w<


total_loss��>


accuracy_1�z?�[,�]       a[��	3n�,_��A�(*O

prediction_loss)\?

reg_loss>:w<


total_loss9?


accuracy_1�G�>fuE,]       a[��	i��,_��A�(*O

prediction_loss���>

reg_loss5:w<


total_lossa|�>


accuracy_1�?��b�]       a[��	���,_��A�(*O

prediction_loss�G�>

reg_loss):w<


total_loss�>


accuracy_1)\?�C�>]       a[��	T�,_��A�(*O

prediction_loss\��>

reg_loss :w<


total_loss-I�>


accuracy_1R�?t��]       a[��	�*�,_��A�(*O

prediction_loss��>

reg_loss:w<


total_loss�>�>


accuracy_1q=
?�פt]       a[��	GT�,_��A�(*O

prediction_loss
ף>

reg_loss:w<


total_lossڐ�>


accuracy_1{.?���]       a[��	�z�,_��A�(*O

prediction_loss��>

reg_loss:w<


total_loss�>�>


accuracy_1q=
?)$m|]       a[��	���,_��A�(*O

prediction_loss�G�>

reg_loss�9w<


total_loss~�>


accuracy_1)\?�2�]       a[��	�н,_��A�(*O

prediction_loss=
�>

reg_loss�9w<


total_loss��>


accuracy_1�z?�[A�]       a[��	v��,_��A�(*O

prediction_loss�z?

reg_loss�9w<


total_loss�W?


accuracy_1=
�>��]       a[��	>�,_��A�(*O

prediction_loss�?

reg_loss�9w<


total_loss��?


accuracy_1���>��tB]       a[��	�p�,_��A�(*O

prediction_loss�?

reg_loss�9w<


total_loss��?


accuracy_1���>�m�']       a[��	��,_��A�(*O

prediction_loss���>

reg_loss�9w<


total_loss]|�>


accuracy_1�?Xd�^]       a[��	"þ,_��A�(*O

prediction_loss)\?

reg_loss�9w<


total_loss9?


accuracy_1�G�>U��]       a[��	��,_��A�(*O

prediction_lossq=
?

reg_loss�9w<


total_lossX?


accuracy_1��>�]       a[��	�,_��A�(*O

prediction_loss�?

reg_loss�9w<


total_loss��?


accuracy_1���>����]       a[��	gG�,_��A�(*O

prediction_loss���>

reg_loss�9w<


total_loss\|�>


accuracy_1�?�1�]       a[��	�s�,_��A�(*O

prediction_loss�G�>

reg_loss�9w<


total_lossz�>


accuracy_1)\?HV]       a[��	ș�,_��A�(*O

prediction_loss   ?

reg_loss�9w<


total_loss��?


accuracy_1   ?o+]       a[��	v¿,_��A�(*O

prediction_lossq=
?

reg_lossx9w<


total_lossW?


accuracy_1��>T,��]       a[��	��,_��A�(*O

prediction_loss�z?

reg_lossn9w<


total_loss�W?


accuracy_1=
�>J�k�]       a[��	B
�,_��A�(*O

prediction_loss�z?

reg_lossd9w<


total_loss�W?


accuracy_1=
�>{>Q]       a[��	EK�,_��A�(*O

prediction_loss   ?

reg_loss[9w<


total_loss��?


accuracy_1   ?.��]       a[��	Do�,_��A�(*O

prediction_loss���>

reg_lossO9w<


total_loss���>


accuracy_1��?� �S]       a[��	���,_��A�(*O

prediction_loss��?

reg_lossE9w<


total_lossv?


accuracy_1���>˃�]       a[��	]��,_��A�(*O

prediction_loss
ף>

reg_loss:9w<


total_lossԐ�>


accuracy_1{.?�ד�]       a[��	�|�,_��A�(*O

prediction_lossq=
?

reg_loss/9w<


total_lossV?


accuracy_1��>o���]       a[��	���,_��A�(*O

prediction_loss
�#?

reg_loss%9w<


total_loss�'?


accuracy_1�Q�>���]       a[��	H��,_��A�(*O

prediction_loss��?

reg_loss9w<


total_loss~v?


accuracy_1���>m}z]       a[��	��,_��A�(*O

prediction_loss��>

reg_loss9w<


total_loss�>�>


accuracy_1q=
?K��C]       a[��	�b�,_��A�(*O

prediction_loss�z?

reg_loss9w<


total_loss�W?


accuracy_1=
�>SP(�]       a[��	���,_��A�(*O

prediction_loss��>

reg_loss�8w<


total_loss�>�>


accuracy_1q=
?�<�]       a[��	ȷ�,_��A�(*O

prediction_lossq=
?

reg_loss�8w<


total_lossU?


accuracy_1��>� )]       a[��	���,_��A�(*O

prediction_loss��>

reg_loss�8w<


total_loss�>�>


accuracy_1q=
?z�]       a[��	��,_��A�(*O

prediction_loss���>

reg_loss�8w<


total_lossV|�>


accuracy_1�?�W�]       a[��	t�,_��A�(*O

prediction_loss��?

reg_loss�8w<


total_loss}v?


accuracy_1���> ��]       a[��	��,_��A�(*O

prediction_loss   ?

reg_loss�8w<


total_loss��?


accuracy_1   ?;�b�]       a[��	? �,_��A�(*O

prediction_loss�z?

reg_loss�8w<


total_loss�W?


accuracy_1=
�>\e[]       a[��	W"�,_��A�(*O

prediction_loss)\�>

reg_loss�8w<


total_loss��>


accuracy_1�Q8?����]       a[��	:\�,_��A�(*O

prediction_lossq=
?

reg_loss�8w<


total_lossT?


accuracy_1��>Xǲ:]       a[��	���,_��A�(*O

prediction_loss���>

reg_loss�8w<


total_lossT|�>


accuracy_1�?���]       a[��	��,_��A�(*O

prediction_lossq=
?

reg_loss�8w<


total_lossS?


accuracy_1��>���%]       a[��	�,_��A�(*O

prediction_loss��>

reg_loss�8w<


total_loss�>�>


accuracy_1q=
?gQ�]       a[��	�w�,_��A�(*O

prediction_loss��>

reg_loss�8w<


total_loss�>�>


accuracy_1q=
?���]       a[��	��,_��A�(*O

prediction_loss�z?

reg_lossw8w<


total_loss�W?


accuracy_1=
�>���]       a[��	 ��,_��A�(*O

prediction_loss�?

reg_lossl8w<


total_loss��?


accuracy_1���>{�Ah]       a[��	L��,_��A�(*O

prediction_loss���>

reg_lossb8w<


total_lossR|�>


accuracy_1�?Q~f"]       a[��	�&�,_��A�(*O

prediction_loss�?

reg_lossX8w<


total_loss��?


accuracy_1���>�iE7]       a[��	Eg�,_��A�(*O

prediction_loss��?

reg_lossN8w<


total_loss{v?


accuracy_1���>�Ez]       a[��	��,_��A�(*O

prediction_loss��>

reg_lossD8w<


total_loss�>�>


accuracy_1q=
?{�]       a[��	���,_��A�(*O

prediction_loss�z?

reg_loss:8w<


total_loss�W?


accuracy_1=
�>d��]       a[��	,�,_��A�(*O

prediction_loss   ?

reg_loss#8w<


total_loss��?


accuracy_1   ?�s|]       a[��	 ^�,_��A�(*O

prediction_loss�z?

reg_loss8w<


total_loss�W?


accuracy_1=
�>�G]       a[��	>��,_��A�(*O

prediction_loss
�#?

reg_loss8w<


total_loss�'?


accuracy_1�Q�>�OJP]       a[��	I��,_��A�(*O

prediction_loss   ?

reg_loss8w<


total_loss��?


accuracy_1   ?��]       a[��	��,_��A�(*O

prediction_loss�z?

reg_loss�7w<


total_loss�W?


accuracy_1=
�>�-L{]       a[��	��,_��A�(*O

prediction_loss   ?

reg_loss�7w<


total_loss��?


accuracy_1   ?ڕMu]       a[��	$B�,_��A�(*O

prediction_loss���>

reg_loss�7w<


total_lossN|�>


accuracy_1�?���]       a[��	rj�,_��A�(*O

prediction_lossq=
?

reg_loss�7w<


total_lossP?


accuracy_1��>���]       a[��	��,_��A�(*O

prediction_loss��?

reg_loss�7w<


total_lossyv?


accuracy_1���>F=�]       a[��	 ��,_��A�(*O

prediction_loss=
�>

reg_loss�7w<


total_loss���>


accuracy_1�z?�=#�]       a[��	>��,_��A�(*O

prediction_loss���>

reg_loss�7w<


total_loss���>


accuracy_1��?�k�d]       a[��	t#�,_��A�(*O

prediction_loss)\?

reg_loss�7w<


total_loss9?


accuracy_1�G�>ޯ��]       a[��	2V�,_��A�(*O

prediction_loss�?

reg_loss�7w<


total_loss��?


accuracy_1���>6���]       a[��	Ȳ�,_��A�(*O

prediction_lossq=
?

reg_loss�7w<


total_lossP?


accuracy_1��>�/�4]       a[��	m6�,_��A�(*O

prediction_loss��>

reg_loss�7w<


total_loss�>�>


accuracy_1q=
?�Rω]       a[��	�i�,_��A�(*O

prediction_lossR�?

reg_loss�7w<


total_loss0�"?


accuracy_1\��>}$�]       a[��	��,_��A�(*O

prediction_loss�?

reg_loss�7w<


total_loss��?


accuracy_1���>����]       a[��	���,_��A�(*O

prediction_loss�z?

reg_lossx7w<


total_loss�W?


accuracy_1=
�>��]       a[��	b�,_��A�(*O

prediction_loss=
�>

reg_lossm7w<


total_loss���>


accuracy_1�z?�,@w]       a[��	�5�,_��A�(*O

prediction_loss���>

reg_lossb7w<


total_loss���>


accuracy_1��?�E]       a[��	�a�,_��A�(*O

prediction_loss=
�>

reg_lossY7w<


total_loss���>


accuracy_1�z?�|{]       a[��	C��,_��A�(*O

prediction_loss�?

reg_lossN7w<


total_loss��?


accuracy_1���>��a9]       a[��	���,_��A�(*O

prediction_loss)\?

reg_lossB7w<


total_loss9?


accuracy_1�G�>���]       a[��	��,_��A�(*O

prediction_loss=
�>

reg_loss:7w<


total_loss���>


accuracy_1�z?³��]       a[��	�H�,_��A�(*O

prediction_loss�z?

reg_loss.7w<


total_loss�W?


accuracy_1=
�>-��]       a[��	�m�,_��A�(*O

prediction_loss�?

reg_loss&7w<


total_loss��?


accuracy_1���>g"b]       a[��	W��,_��A�(*O

prediction_loss���>

reg_loss7w<


total_lossH|�>


accuracy_1�?p8��]       a[��	N��,_��A�(*O

prediction_lossR�?

reg_loss7w<


total_loss.�"?


accuracy_1\��>���]       a[��	D��,_��A�(*O

prediction_loss��?

reg_loss7w<


total_lossvv?


accuracy_1���>���]       a[��	�(�,_��A�(*O

prediction_lossq=
?

reg_loss�6w<


total_lossM?


accuracy_1��>9���]       a[��	�K�,_��A�(*O

prediction_loss��>

reg_loss�6w<


total_loss�>�>


accuracy_1q=
?k��]       a[��	>{�,_��A�(*O

prediction_loss)\?

reg_loss�6w<


total_loss9?


accuracy_1�G�>���s]       a[��	��,_��A�(*O

prediction_loss�z?

reg_loss�6w<


total_loss�W?


accuracy_1=
�>K���]       a[��	!��,_��A�(*O

prediction_lossq=
?

reg_loss�6w<


total_lossL?


accuracy_1��>��<	]       a[��	�
�,_��A�(*O

prediction_loss��?

reg_loss�6w<


total_lossuv?


accuracy_1���>��s5]       a[��	U4�,_��A�(*O

prediction_loss=
�>

reg_loss�6w<


total_loss���>


accuracy_1�z?�!��]       a[��	�z�,_��A�(*O

prediction_loss��?

reg_loss�6w<


total_lossuv?


accuracy_1���>(z�g]       a[��	���,_��A�(*O

prediction_loss���>

reg_loss�6w<


total_loss���>


accuracy_1��?���@]       a[��	���,_��A�(*O

prediction_loss��?

reg_loss�6w<


total_lossuv?


accuracy_1���>����]       a[��	S�,_��A�(*O

prediction_loss���>

reg_loss�6w<


total_lossD|�>


accuracy_1�?��s]       a[��	L8�,_��A�(*O

prediction_loss�z?

reg_loss�6w<


total_loss�W?


accuracy_1=
�>���]       a[��	�a�,_��A�(*O

prediction_loss�Q�>

reg_loss�6w<


total_loss��>


accuracy_1
�#?�S�]       a[��	z��,_��A�(*O

prediction_loss��?

reg_lossz6w<


total_losstv?


accuracy_1���>x��]       a[��	���,_��A�(*O

prediction_lossq=
?

reg_lossl6w<


total_lossK?


accuracy_1��>ꁨ�]       a[��	���,_��A�(*O

prediction_lossq=
?

reg_lossd6w<


total_lossK?


accuracy_1��>�T�W]       a[��	��,_��A�(*O

prediction_loss\��>

reg_lossY6w<


total_lossI�>


accuracy_1R�?_9T]       a[��	�:�,_��A�(*O

prediction_loss���>

reg_lossO6w<


total_lossA|�>


accuracy_1�?l��]]       a[��	�d�,_��A�)*O

prediction_loss��>

reg_lossE6w<


total_loss�>�>


accuracy_1q=
?�m�y]       a[��	���,_��A�)*O

prediction_loss=
�>

reg_loss:6w<


total_loss���>


accuracy_1�z?Y�Ƶ]       a[��	c��,_��A�)*O

prediction_loss���>

reg_loss/6w<


total_loss~��>


accuracy_1��?��#]       a[��	���,_��A�)*O

prediction_loss��>

reg_loss&6w<


total_loss�>�>


accuracy_1q=
?Cr��]       a[��	*�,_��A�)*O

prediction_loss���>

reg_loss6w<


total_loss@|�>


accuracy_1�?�eg*]       a[��	�(�,_��A�)*O

prediction_loss��>

reg_loss6w<


total_loss�>�>


accuracy_1q=
?�]       a[��	�P�,_��A�)*O

prediction_lossq=
?

reg_loss6w<


total_lossI?


accuracy_1��>m��]       a[��	��,_��A�)*O

prediction_loss   ?

reg_loss�5w<


total_loss��?


accuracy_1   ?:N��]       a[��	���,_��A�)*O

prediction_loss�?

reg_loss�5w<


total_loss��?


accuracy_1���>"J]       a[��	�,_��A�)*O

prediction_loss=
�>

reg_loss�5w<


total_loss���>


accuracy_1�z?��!]       a[��	�1�,_��A�)*O

prediction_loss��>

reg_loss�5w<


total_loss�>�>


accuracy_1q=
?h��]       a[��	U�,_��A�)*O

prediction_loss�G�>

reg_loss�5w<


total_loss]�>


accuracy_1)\?N	dA]       a[��	Oy�,_��A�)*O

prediction_loss=
�>

reg_loss�5w<


total_loss���>


accuracy_1�z?�{��]       a[��	��,_��A�)*O

prediction_loss��>

reg_loss�5w<


total_loss�>�>


accuracy_1q=
?���]       a[��	��,_��A�)*O

prediction_loss���>

reg_loss�5w<


total_loss=|�>


accuracy_1�?��]       a[��	���,_��A�)*O

prediction_loss��>

reg_loss�5w<


total_loss�>�>


accuracy_1q=
?��� ]       a[��	��,_��A�)*O

prediction_lossq=
?

reg_loss�5w<


total_lossH?


accuracy_1��>�U�]       a[��	�>�,_��A�)*O

prediction_loss�?

reg_loss�5w<


total_loss��?


accuracy_1���>�Q)]       a[��	,b�,_��A�)*O

prediction_loss)\?

reg_loss�5w<


total_loss�8?


accuracy_1�G�>K���]       a[��	O��,_��A�)*O

prediction_loss   ?

reg_loss�5w<


total_loss��?


accuracy_1   ?�W4�]       a[��	=��,_��A�)*O

prediction_loss�?

reg_lossx5w<


total_loss��?


accuracy_1���>6��]       a[��	���,_��A�)*O

prediction_loss�?

reg_lossl5w<


total_loss��?


accuracy_1���>Ӣ#]       a[��	_�,_��A�)*O

prediction_lossq=
?

reg_lossd5w<


total_lossG?


accuracy_1��>���
]       a[��	�1�,_��A�)*O

prediction_loss���>

reg_lossY5w<


total_loss:|�>


accuracy_1�?�QF�]       a[��	�[�,_��A�)*O

prediction_loss��>

reg_lossP5w<


total_loss�>�>


accuracy_1q=
?%x]       a[��	��,_��A�)*O

prediction_loss�?

reg_lossC5w<


total_loss��?


accuracy_1���>�$�>]       a[��	+��,_��A�)*O

prediction_loss���>

reg_loss:5w<


total_loss9|�>


accuracy_1�?m�Lx]       a[��	���,_��A�)*O

prediction_loss�G�>

reg_loss/5w<


total_lossW�>


accuracy_1)\?�U��]       a[��	���,_��A�)*O

prediction_loss
�#?

reg_loss%5w<


total_loss߳'?


accuracy_1�Q�>Բ&�]       a[��	��,_��A�)*O

prediction_loss���>

reg_loss5w<


total_loss8|�>


accuracy_1�?���{]       a[��	�>�,_��A�)*O

prediction_loss�?

reg_loss5w<


total_loss��?


accuracy_1���>��]       a[��	�n�,_��A�)*O

prediction_loss�G�>

reg_loss5w<


total_lossV�>


accuracy_1)\?d�]�]       a[��	���,_��A�)*O

prediction_loss��>

reg_loss�4w<


total_loss�>�>


accuracy_1q=
?�\�]       a[��	o��,_��A�)*O

prediction_lossq=
?

reg_loss�4w<


total_lossE?


accuracy_1��>FSU]       a[��	d��,_��A�)*O

prediction_loss�?

reg_loss�4w<


total_loss��?


accuracy_1���>uBd]       a[��	h�,_��A�)*O

prediction_loss�G�>

reg_loss�4w<


total_lossU�>


accuracy_1)\?�vA�]       a[��	8�,_��A�)*O

prediction_loss�G�>

reg_loss�4w<


total_lossU�>


accuracy_1)\?��z]       a[��	�b�,_��A�)*O

prediction_loss   ?

reg_loss�4w<


total_loss��?


accuracy_1   ?��ƀ]       a[��	��,_��A�)*O

prediction_loss)\?

reg_loss�4w<


total_loss�8?


accuracy_1�G�>V�]       a[��	���,_��A�)*O

prediction_loss�?

reg_loss�4w<


total_loss��?


accuracy_1���>W��P]       a[��	���,_��A�)*O

prediction_loss)\?

reg_loss�4w<


total_loss�8?


accuracy_1�G�>k�]       a[��	��,_��A�)*O

prediction_loss��>

reg_loss�4w<


total_loss�>�>


accuracy_1q=
?�(4]       a[��	a3�,_��A�)*O

prediction_loss{�>

reg_loss�4w<


total_loss ε>


accuracy_1��(?^ �]       a[��	T�,_��A�)*O

prediction_lossR�?

reg_loss�4w<


total_loss$�"?


accuracy_1\��>{մ]       a[��	 }�,_��A�)*O

prediction_loss)\?

reg_loss�4w<


total_loss�8?


accuracy_1�G�>c�U]       a[��	\��,_��A�)*O

prediction_loss��(?

reg_lossw4w<


total_loss��,?


accuracy_1{�>�!�]       a[��	5��,_��A�)*O

prediction_loss��>

reg_lossk4w<


total_loss�>�>


accuracy_1q=
?���%]       a[��	i��,_��A�)*O

prediction_loss�z?

reg_lossb4w<


total_loss�W?


accuracy_1=
�>~B|]       a[��	�4�,_��A�)*O

prediction_loss)\?

reg_lossW4w<


total_loss�8?


accuracy_1�G�>#hY�]       a[��	h^�,_��A�)*O

prediction_loss��>

reg_lossL4w<


total_loss�>�>


accuracy_1q=
?�/]       a[��		��,_��A�)*O

prediction_lossq=
?

reg_lossB4w<


total_lossB?


accuracy_1��>9B�]       a[��	���,_��A�)*O

prediction_loss���>

reg_loss84w<


total_losso��>


accuracy_1��?rи']       a[��	/��,_��A�)*O

prediction_loss)\?

reg_loss.4w<


total_loss�8?


accuracy_1�G�>��n\]       a[��	+Q�,_��A�)*O

prediction_loss�G�>

reg_loss4w<


total_lossO�>


accuracy_1)\?��G]       a[��	��,_��A�)*O

prediction_loss���>

reg_loss4w<


total_loss/|�>


accuracy_1�?*u�]       a[��	X��,_��A�)*O

prediction_loss
�#?

reg_loss4w<


total_lossڳ'?


accuracy_1�Q�>���]       a[��	���,_��A�)*O

prediction_loss��>

reg_loss�3w<


total_loss�>�>


accuracy_1q=
?�ه]       a[��	8�,_��A�)*O

prediction_loss)\?

reg_loss�3w<


total_loss�8?


accuracy_1�G�>��]       a[��	m9�,_��A�)*O

prediction_loss��?

reg_loss�3w<


total_lossjv?


accuracy_1���>��_]       a[��	�Y�,_��A�)*O

prediction_loss�Q�>

reg_loss�3w<


total_loss��>


accuracy_1
�#?��e�]       a[��	��,_��A�)*O

prediction_loss�G�>

reg_loss�3w<


total_lossL�>


accuracy_1)\?�s[�]       a[��	���,_��A�)*O

prediction_loss���>

reg_loss�3w<


total_lossk��>


accuracy_1��?:2D]       a[��	/��,_��A�)*O

prediction_loss�Q�>

reg_loss�3w<


total_loss��>


accuracy_1
�#?u#�
]       a[��	�
�,_��A�)*O

prediction_loss)\?

reg_loss�3w<


total_loss�8?


accuracy_1�G�>\�D�]       a[��	�7�,_��A�)*O

prediction_loss�?

reg_loss�3w<


total_loss��?


accuracy_1���>��]       a[��	�e�,_��A�)*O

prediction_loss���>

reg_loss�3w<


total_loss,|�>


accuracy_1�?��/�]       a[��	��,_��A�)*O

prediction_loss���>

reg_loss�3w<


total_loss,|�>


accuracy_1�?X�B�]       a[��	?��,_��A�)*O

prediction_loss)\?

reg_loss�3w<


total_loss�8?


accuracy_1�G�>xW�:]       a[��	R)�,_��A�)*O

prediction_loss�?

reg_loss}3w<


total_loss��?


accuracy_1���>��m{]       a[��	7T�,_��A�)*O

prediction_loss�G�>

reg_lossr3w<


total_lossJ�>


accuracy_1)\?��d�]       a[��	G��,_��A�)*O

prediction_loss���>

reg_lossi3w<


total_loss*|�>


accuracy_1�?�k��]       a[��	���,_��A�)*O

prediction_loss��>

reg_loss]3w<


total_loss�>�>


accuracy_1q=
?,wٗ]       a[��	���,_��A�)*O

prediction_lossq=
?

reg_lossU3w<


total_loss>?


accuracy_1��>���+]       a[��	~�,_��A�)*O

prediction_loss���>

reg_lossJ3w<


total_lossg��>


accuracy_1��?O��]       a[��	�D�,_��A�)*O

prediction_loss)\?

reg_lossA3w<


total_loss�8?


accuracy_1�G�>��M�]       a[��	�n�,_��A�)*O

prediction_loss���>

reg_loss43w<


total_loss)|�>


accuracy_1�?�q]       a[��	Ֆ�,_��A�)*O

prediction_loss\��>

reg_loss*3w<


total_loss�H�>


accuracy_1R�?��V�]       a[��	��,_��A�)*O

prediction_loss���>

reg_loss 3w<


total_loss(|�>


accuracy_1�?�X[�]       a[��	���,_��A�)*O

prediction_loss   ?

reg_loss3w<


total_loss��?


accuracy_1   ?���]       a[��	��,_��A�)*O

prediction_loss��?

reg_loss3w<


total_lossfv?


accuracy_1���>�Q]       a[��	P5�,_��A�)*O

prediction_loss)\?

reg_loss3w<


total_loss�8?


accuracy_1�G�>���]       a[��	���,_��A�)*O

prediction_loss=
�>

reg_loss�2w<


total_loss���>


accuracy_1�z?���D]       a[��	1	�,_��A�)*O

prediction_lossq=
?

reg_loss�2w<


total_loss=?


accuracy_1��>�F]       a[��	3�,_��A�)*O

prediction_loss���>

reg_loss�2w<


total_loss&|�>


accuracy_1�?��a�]       a[��	uV�,_��A�)*O

prediction_loss�z?

reg_loss�2w<


total_loss�W?


accuracy_1=
�>�*��]       a[��	�|�,_��A�)*O

prediction_loss���>

reg_loss�2w<


total_loss%|�>


accuracy_1�?hȏ�]       a[��	��,_��A�)*O

prediction_lossq=
?

reg_loss�2w<


total_loss<?


accuracy_1��>����]       a[��	O��,_��A�)*O

prediction_loss�z?

reg_loss�2w<


total_loss�W?


accuracy_1=
�>�j�W]       a[��	b��,_��A�)*O

prediction_lossq=
?

reg_loss�2w<


total_loss<?


accuracy_1��>w�]       a[��	��,_��A�)*O

prediction_loss=
�>

reg_loss�2w<


total_loss���>


accuracy_1�z?���]       a[��	[C�,_��A�)*O

prediction_lossq=
?

reg_loss�2w<


total_loss;?


accuracy_1��>���]       a[��	3o�,_��A�)*O

prediction_loss�G�>

reg_loss�2w<


total_lossB�>


accuracy_1)\?�_,�]       a[��	���,_��A�)*O

prediction_loss   ?

reg_loss�2w<


total_loss��?


accuracy_1   ?�H�]       a[��	[��,_��A�)*O

prediction_loss�?

reg_lossz2w<


total_loss��?


accuracy_1���>��X�]       a[��	���,_��A�)*O

prediction_loss��>

reg_lossp2w<


total_loss�>�>


accuracy_1q=
?=�gq]       a[��	��,_��A�)*O

prediction_loss   ?

reg_lossf2w<


total_loss��?


accuracy_1   ?���p]       a[��	�.�,_��A�)*O

prediction_loss�?

reg_loss[2w<


total_loss��?


accuracy_1���>�]       a[��	�P�,_��A�)*O

prediction_loss�?

reg_lossR2w<


total_loss��?


accuracy_1���>^]       a[��	zr�,_��A�)*O

prediction_loss   ?

reg_lossH2w<


total_loss��?


accuracy_1   ?p�]       a[��	���,_��A�)*O

prediction_lossq=
?

reg_loss<2w<


total_loss:?


accuracy_1��>�sN]       a[��	Ե�,_��A�)*O

prediction_loss�?

reg_loss22w<


total_loss��?


accuracy_1���>�N�]       a[��	M��,_��A�)*O

prediction_loss�z?

reg_loss)2w<


total_loss�W?


accuracy_1=
�>�N�u]       a[��	���,_��A�)*O

prediction_loss)\?

reg_loss2w<


total_loss�8?


accuracy_1�G�>g�B�]       a[��	x�,_��A�)*O

prediction_loss��>

reg_loss2w<


total_loss�>�>


accuracy_1q=
?�-��]       a[��	�2�,_��A�)*O

prediction_loss�z?

reg_loss
2w<


total_loss�W?


accuracy_1=
�>�]A]       a[��	aQ�,_��A�)*O

prediction_lossR�?

reg_loss2w<


total_loss�"?


accuracy_1\��>� Y�]       a[��	`t�,_��A�)*O

prediction_loss���>

reg_loss�1w<


total_loss|�>


accuracy_1�?�i]       a[��	)��,_��A�)*O

prediction_loss   ?

reg_loss�1w<


total_loss��?


accuracy_1   ?�h�2]       a[��	ȴ�,_��A�)*O

prediction_loss=
�>

reg_loss�1w<


total_loss���>


accuracy_1�z?O���]       a[��	���,_��A�)*O

prediction_loss���>

reg_loss�1w<


total_loss|�>


accuracy_1�?_��]       a[��	��,_��A�)*O

prediction_loss�z?

reg_loss�1w<


total_loss�W?


accuracy_1=
�>flK-]       a[��	x&�,_��A�)*O

prediction_loss)\?

reg_loss�1w<


total_loss�8?


accuracy_1�G�>����]       a[��	�J�,_��A�)*O

prediction_lossR�?

reg_loss�1w<


total_loss�"?


accuracy_1\��>꣩]       a[��	�h�,_��A�)*O

prediction_loss)\?

reg_loss�1w<


total_loss�8?


accuracy_1�G�>
��]       a[��	j��,_��A�)*O

prediction_lossq=
?

reg_loss�1w<


total_loss8?


accuracy_1��>�A�]       a[��	Q��,_��A�)*O

prediction_loss���>

reg_loss�1w<


total_loss|�>


accuracy_1�?l�O]       a[��	���,_��A�)*O

prediction_loss��(?

reg_loss�1w<


total_loss��,?


accuracy_1{�>�S.]       a[��	(��,_��A�)*O

prediction_loss�z?

reg_loss�1w<


total_loss�W?


accuracy_1=
�>]\R]       a[��	��,_��A�)*O

prediction_loss\��>

reg_loss{1w<


total_loss�H�>


accuracy_1R�?���]       a[��	�.�,_��A�)*O

prediction_loss���>

reg_lossp1w<


total_lossX��>


accuracy_1��?op��]       a[��	�V�,_��A�)*O

prediction_loss)\?

reg_losse1w<


total_loss�8?


accuracy_1�G�>�i͒]       a[��	Ow�,_��A�)*O

prediction_loss���>

reg_loss[1w<


total_loss|�>


accuracy_1�?�U��]       a[��	-��,_��A�)*O

prediction_loss�G�>

reg_lossQ1w<


total_loss9�>


accuracy_1)\?�2@>]       a[��	˹�,_��A�)*O

prediction_loss�z?

reg_lossF1w<


total_loss�W?


accuracy_1=
�> �	�]       a[��	���,_��A�)*O

prediction_loss�z?

reg_loss<1w<


total_loss�W?


accuracy_1=
�>�A5]       a[��	��,_��A�)*O

prediction_lossR�?

reg_loss21w<


total_loss�"?


accuracy_1\��>��
�]       a[��	�[�,_��A�)*O

prediction_loss���>

reg_loss(1w<


total_loss|�>


accuracy_1�?��1�]       a[��	�}�,_��A�)*O

prediction_loss�Q�>

reg_loss1w<


total_lossu�>


accuracy_1
�#?��v/]       a[��	ҧ�,_��A�**O

prediction_loss���>

reg_loss1w<


total_lossV��>


accuracy_1��?F��+]       a[��	(��,_��A�**O

prediction_loss   ?

reg_loss1w<


total_loss��?


accuracy_1   ?�:?=]       a[��	��,_��A�**O

prediction_loss�z?

reg_loss�0w<


total_loss�W?


accuracy_1=
�>E~/�]       a[��	�B�,_��A�**O

prediction_loss���>

reg_loss�0w<


total_loss|�>


accuracy_1�?�׌]       a[��	qr�,_��A�**O

prediction_loss��>

reg_loss�0w<


total_loss�>�>


accuracy_1q=
?�d�]       a[��	B��,_��A�**O

prediction_loss   ?

reg_loss�0w<


total_loss��?


accuracy_1   ?��~�]       a[��	���,_��A�**O

prediction_loss
�#?

reg_loss�0w<


total_lossͳ'?


accuracy_1�Q�>&�R]       a[��	q��,_��A�**O

prediction_loss��>

reg_loss�0w<


total_loss�>�>


accuracy_1q=
?u���]       a[��	��,_��A�**O

prediction_loss�G�>

reg_loss�0w<


total_loss4�>


accuracy_1)\?Ҏ;e]       a[��	28�,_��A�**O

prediction_loss�Q8?

reg_loss�0w<


total_loss�.<?


accuracy_1)\�>w,�]       a[��	�W�,_��A�**O

prediction_lossq=
?

reg_loss�0w<


total_loss4?


accuracy_1��>Ua��]       a[��	�y�,_��A�**O

prediction_loss=
�>

reg_loss�0w<


total_loss���>


accuracy_1�z?oN��]       a[��	˝�,_��A�**O

prediction_loss���>

reg_loss�0w<


total_lossR��>


accuracy_1��?b?�V]       a[��	���,_��A�**O

prediction_loss�?

reg_loss�0w<


total_lossz�?


accuracy_1���>�]       a[��	���,_��A�**O

prediction_loss�?

reg_loss�0w<


total_lossz�?


accuracy_1���>^!]       a[��	�,_��A�**O

prediction_loss�G�>

reg_loss{0w<


total_loss2�>


accuracy_1)\?�o]       a[��	Y1�,_��A�**O

prediction_lossq=
?

reg_lossn0w<


total_loss3?


accuracy_1��>XM�j]       a[��	nP�,_��A�**O

prediction_loss   ?

reg_lossf0w<


total_loss��?


accuracy_1   ?���}]       a[��	J}�,_��A�**O

prediction_loss\��>

reg_lossZ0w<


total_loss�H�>


accuracy_1R�?��@�]       a[��	j��,_��A�**O

prediction_loss�?

reg_lossQ0w<


total_lossy�?


accuracy_1���>LM�m]       a[��	���,_��A�**O

prediction_loss�G�>

reg_lossG0w<


total_loss0�>


accuracy_1)\?��fO]       a[��	��,_��A�**O

prediction_loss)\?

reg_loss=0w<


total_loss�8?


accuracy_1�G�>�>��]       a[��	��,_��A�**O

prediction_loss=
�>

reg_loss10w<


total_loss���>


accuracy_1�z?���q]       a[��	�8�,_��A�**O

prediction_loss��>

reg_loss'0w<


total_loss�>�>


accuracy_1q=
?~0\]       a[��	�N�,_��A�**O

prediction_loss��>

reg_loss0w<


total_loss�>�>


accuracy_1q=
?|�1C]       a[��	U��,_��A�**O

prediction_loss���>

reg_loss	0w<


total_loss|�>


accuracy_1�?����]       a[��	���,_��A�**O

prediction_loss�G�>

reg_loss�/w<


total_loss.�>


accuracy_1)\?���j]       a[��	��,_��A�**O

prediction_loss   ?

reg_loss�/w<


total_loss��?


accuracy_1   ?���]       a[��	-�,_��A�**O

prediction_loss=
�>

reg_loss�/w<


total_loss���>


accuracy_1�z?7>_]       a[��	bI�,_��A�**O

prediction_lossR�?

reg_loss�/w<


total_loss�"?


accuracy_1\��>@���]       a[��	�^�,_��A�**O

prediction_loss���>

reg_loss�/w<


total_lossL��>


accuracy_1��?�Ŗ{]       a[��	i��,_��A�**O

prediction_loss�?

reg_loss�/w<


total_lossw�?


accuracy_1���>�T��]       a[��	���,_��A�**O

prediction_loss��>

reg_loss�/w<


total_loss�>�>


accuracy_1q=
?s�	]       a[��	 ��,_��A�**O

prediction_loss��>

reg_loss�/w<


total_loss�>�>


accuracy_1q=
?���]       a[��	�,_��A�**O

prediction_lossq=
?

reg_loss�/w<


total_loss0?


accuracy_1��>耢�]       a[��	B&�,_��A�**O

prediction_loss�z?

reg_loss�/w<


total_loss�W?


accuracy_1=
�>'�SE]       a[��	�B�,_��A�**O

prediction_loss)\?

reg_loss�/w<


total_loss�8?


accuracy_1�G�>d68]       a[��	1[�,_��A�**O

prediction_loss�G�>

reg_loss�/w<


total_loss*�>


accuracy_1)\?:͚�]       a[��	Fy�,_��A�**O

prediction_loss��>

reg_loss�/w<


total_loss�>�>


accuracy_1q=
?�t]       a[��	��,_��A�**O

prediction_loss{.?

reg_lossy/w<


total_loss9�1?


accuracy_1
ף>���]       a[��	��,_��A�**O

prediction_loss�z?

reg_losso/w<


total_loss�W?


accuracy_1=
�>@Շ�]       a[��	M��,_��A�**O

prediction_loss�?

reg_lossf/w<


total_lossv�?


accuracy_1���>؆��]       a[��	��,_��A�**O

prediction_loss{�>

reg_lossY/w<


total_loss�͵>


accuracy_1��(?�ڣ]       a[��	W	�,_��A�**O

prediction_loss�z?

reg_lossQ/w<


total_loss�W?


accuracy_1=
�>�H�)]       a[��	W&�,_��A�**O

prediction_loss   ?

reg_lossG/w<


total_loss��?


accuracy_1   ?���]       a[��	AF�,_��A�**O

prediction_loss���>

reg_loss;/w<


total_loss	|�>


accuracy_1�?��!�]       a[��	�a�,_��A�**O

prediction_loss��>

reg_loss0/w<


total_loss�>�>


accuracy_1q=
?�R��]       a[��	��,_��A�**O

prediction_lossq=
?

reg_loss'/w<


total_loss.?


accuracy_1��>��]       a[��	*��,_��A�**O

prediction_loss   ?

reg_loss/w<


total_loss��?


accuracy_1   ?�:]       a[��	���,_��A�**O

prediction_loss���>

reg_loss/w<


total_loss|�>


accuracy_1�?�K�j]       a[��	(��,_��A�**O

prediction_loss�z?

reg_loss/w<


total_loss�W?


accuracy_1=
�>~���]       a[��	��,_��A�**O

prediction_loss�G�>

reg_loss�.w<


total_loss&�>


accuracy_1)\?�׮�]       a[��	�2�,_��A�**O

prediction_loss   ?

reg_loss�.w<


total_loss��?


accuracy_1   ?�[ؘ]       a[��	�[�,_��A�**O

prediction_loss�z?

reg_loss�.w<


total_loss�W?


accuracy_1=
�>4c��]       a[��	�}�,_��A�**O

prediction_loss   ?

reg_loss�.w<


total_loss��?


accuracy_1   ?�I�8]       a[��	Q��,_��A�**O

prediction_loss)\?

reg_loss�.w<


total_loss�8?


accuracy_1�G�>�B��]       a[��	û�,_��A�**O

prediction_loss   ?

reg_loss�.w<


total_loss��?


accuracy_1   ?��{�]       a[��	���,_��A�**O

prediction_loss   ?

reg_loss�.w<


total_loss��?


accuracy_1   ?}�Q]       a[��	���,_��A�**O

prediction_loss=
�>

reg_loss�.w<


total_loss���>


accuracy_1�z?PFJ�]       a[��	� �,_��A�**O

prediction_loss
�#?

reg_loss�.w<


total_lossų'?


accuracy_1�Q�>�f��]       a[��	�<�,_��A�**O

prediction_loss   ?

reg_loss�.w<


total_loss��?


accuracy_1   ?m��]       a[��	CV�,_��A�**O

prediction_loss�z?

reg_loss�.w<


total_loss�W?


accuracy_1=
�>��]       a[��	�r�,_��A�**O

prediction_loss
�#?

reg_loss�.w<


total_lossĳ'?


accuracy_1�Q�>�oi�]       a[��	d��,_��A�**O

prediction_loss��>

reg_loss�.w<


total_loss�>�>


accuracy_1q=
?����]       a[��	%��,_��A�**O

prediction_loss\��>

reg_lossz.w<


total_loss�H�>


accuracy_1R�?�Y]       a[��	u��,_��A�**O

prediction_loss��>

reg_losso.w<


total_loss�>�>


accuracy_1q=
?􎥎]       a[��	+��,_��A�**O

prediction_loss���>

reg_lossd.w<


total_loss|�>


accuracy_1�?3��]       a[��	��,_��A�**O

prediction_loss)\?

reg_loss[.w<


total_loss�8?


accuracy_1�G�>��5]       a[��	�=�,_��A�**O

prediction_loss�z?

reg_lossP.w<


total_loss�W?


accuracy_1=
�>�P��]       a[��	uX�,_��A�**O

prediction_loss333?

reg_lossE.w<


total_loss�7?


accuracy_1���>M�ߖ]       a[��	s�,_��A�**O

prediction_lossq=
?

reg_loss:.w<


total_loss*?


accuracy_1��>d˙]       a[��	Ƥ�,_��A�**O

prediction_loss
�#?

reg_loss0.w<


total_lossó'?


accuracy_1�Q�>	.�]]       a[��	5��,_��A�**O

prediction_loss��>

reg_loss'.w<


total_loss�>�>


accuracy_1q=
?	��]       a[��	���,_��A�**O

prediction_loss�G�>

reg_loss.w<


total_loss�>


accuracy_1)\?��A]       a[��	<�,_��A�**O

prediction_loss=
�>

reg_loss.w<


total_loss���>


accuracy_1�z?��ߏ]       a[��	:�,_��A�**O

prediction_loss�Q�>

reg_loss.w<


total_loss\�>


accuracy_1
�#?��]       a[��	�Q�,_��A�**O

prediction_loss   ?

reg_loss�-w<


total_loss��?


accuracy_1   ?��5�]       a[��	Tn�,_��A�**O

prediction_lossR�?

reg_loss�-w<


total_loss
�"?


accuracy_1\��>�0J�]       a[��	 ��,_��A�**O

prediction_loss)\?

reg_loss�-w<


total_loss�8?


accuracy_1�G�>%/<�]       a[��	N��,_��A�**O

prediction_loss��>

reg_loss�-w<


total_loss�>�>


accuracy_1q=
?�I:]       a[��	���,_��A�**O

prediction_loss��?

reg_loss�-w<


total_lossQv?


accuracy_1���>�p��]       a[��	���,_��A�**O

prediction_loss���>

reg_loss�-w<


total_loss�{�>


accuracy_1�?f��]       a[��	��,_��A�**O

prediction_loss�G�>

reg_loss�-w<


total_loss�>


accuracy_1)\?$u(�]       a[��	�2�,_��A�**O

prediction_loss\��>

reg_loss�-w<


total_loss�H�>


accuracy_1R�?��e]       a[��	�T�,_��A�**O

prediction_loss��>

reg_loss�-w<


total_loss�>�>


accuracy_1q=
?�a�d]       a[��	Dm�,_��A�**O

prediction_lossq=
?

reg_loss�-w<


total_loss'?


accuracy_1��>��x]       a[��	+��,_��A�**O

prediction_loss��>

reg_loss�-w<


total_loss�>�>


accuracy_1q=
?o>p�]       a[��	j��,_��A�**O

prediction_loss��?

reg_loss�-w<


total_lossPv?


accuracy_1���>�\��]       a[��	��,_��A�**O

prediction_loss)\?

reg_loss�-w<


total_loss�8?


accuracy_1�G�>]CF]       a[��	K��,_��A�**O

prediction_lossq=
?

reg_lossu-w<


total_loss'?


accuracy_1��>�kh]       a[��	��,_��A�**O

prediction_loss���>

reg_lossk-w<


total_loss�{�>


accuracy_1�?X;k]       a[��	H�,_��A�**O

prediction_loss���>

reg_loss`-w<


total_loss8��>


accuracy_1��?��ٮ]       a[��	�:�,_��A�**O

prediction_loss���>

reg_lossU-w<


total_loss�{�>


accuracy_1�?"�va]       a[��	xc�,_��A�**O

prediction_loss���>

reg_lossK-w<


total_lossS�>


accuracy_1333?5S#]       a[��	=��,_��A�**O

prediction_loss�G�>

reg_lossB-w<


total_loss�>


accuracy_1)\?I3{]       a[��	���,_��A�**O

prediction_loss�?

reg_loss7-w<


total_lossm�?


accuracy_1���>�3�]       a[��	��,_��A�**O

prediction_loss��?

reg_loss+-w<


total_lossOv?


accuracy_1���>ɍL�]       a[��	���,_��A�**O

prediction_lossR�?

reg_loss-w<


total_loss�"?


accuracy_1\��>M��]       a[��	f��,_��A�**O

prediction_loss)\?

reg_loss-w<


total_loss�8?


accuracy_1�G�>�B�G]       a[��	��,_��A�**O

prediction_loss)\?

reg_loss-w<


total_loss�8?


accuracy_1�G�>y�V�]       a[��	�.�,_��A�**O

prediction_loss�?

reg_loss-w<


total_lossl�?


accuracy_1���>֊ۘ]       a[��	�S�,_��A�**O

prediction_lossq=
?

reg_loss�,w<


total_loss%?


accuracy_1��>�ci�]       a[��	�s�,_��A�**O

prediction_loss   ?

reg_loss�,w<


total_loss��?


accuracy_1   ?�;Ao]       a[��	��,_��A�**O

prediction_lossq=
?

reg_loss�,w<


total_loss%?


accuracy_1��>��g5]       a[��	Ϊ�,_��A�**O

prediction_loss=
�>

reg_loss�,w<


total_loss���>


accuracy_1�z?[wB]       a[��	S��,_��A�**O

prediction_loss   ?

reg_loss�,w<


total_loss��?


accuracy_1   ?c��]       a[��	x��,_��A�**O

prediction_loss���>

reg_loss�,w<


total_loss3��>


accuracy_1��?"�~]       a[��	
�,_��A�**O

prediction_loss)\?

reg_loss�,w<


total_loss�8?


accuracy_1�G�>}%�]       a[��	�#�,_��A�**O

prediction_loss   ?

reg_loss�,w<


total_loss��?


accuracy_1   ?i��]       a[��	F�,_��A�**O

prediction_loss   ?

reg_loss�,w<


total_loss��?


accuracy_1   ?W�
P]       a[��	�j�,_��A�**O

prediction_loss�?

reg_loss�,w<


total_lossj�?


accuracy_1���>�)�]       a[��	��,_��A�**O

prediction_lossq=
?

reg_loss�,w<


total_loss#?


accuracy_1��>kZ�]       a[��	��,_��A�**O

prediction_lossq=
?

reg_loss�,w<


total_loss#?


accuracy_1��>�5K]       a[��	���,_��A�**O

prediction_loss)\?

reg_loss~,w<


total_loss�8?


accuracy_1�G�>�R]       a[��	���,_��A�**O

prediction_loss�G�>

reg_losss,w<


total_loss�>


accuracy_1)\?�_#A]       a[��	���,_��A�**O

prediction_loss�?

reg_lossj,w<


total_lossj�?


accuracy_1���>6��`]       a[��	��,_��A�**O

prediction_loss��>

reg_loss_,w<


total_loss�>�>


accuracy_1q=
?�eH]       a[��	�5�,_��A�**O

prediction_loss�G�>

reg_lossT,w<


total_loss�>


accuracy_1)\?r<-]       a[��	�W�,_��A�**O

prediction_loss��(?

reg_lossJ,w<


total_losst�,?


accuracy_1{�>����]       a[��	4��,_��A�**O

prediction_loss��?

reg_lossA,w<


total_lossKv?


accuracy_1���>|��]       a[��	#��,_��A�**O

prediction_loss��?

reg_loss6,w<


total_lossKv?


accuracy_1���>?du]       a[��	���,_��A�**O

prediction_loss���>

reg_loss,,w<


total_loss�{�>


accuracy_1�? 	�]       a[��	��,_��A�**O

prediction_loss�?

reg_loss#,w<


total_lossi�?


accuracy_1���>W��L]       a[��	���,_��A�**O

prediction_loss�G�>

reg_loss,w<


total_loss�>


accuracy_1)\?�']       a[��	��,_��A�**O

prediction_loss�z?

reg_loss,w<


total_loss�W?


accuracy_1=
�>�O�[]       a[��	ǃ�,_��A�**O

prediction_loss   ?

reg_loss�+w<


total_loss��?


accuracy_1   ?�8�d]       a[��	Υ�,_��A�**O

prediction_loss�z?

reg_loss�+w<


total_loss�W?


accuracy_1=
�>k�E]       a[��	���,_��A�**O

prediction_loss�G�>

reg_loss�+w<


total_loss�>


accuracy_1)\?��,�]       a[��	���,_��A�+*O

prediction_loss�G�>

reg_loss�+w<


total_loss�>


accuracy_1)\?�ލ�]       a[��	G��,_��A�+*O

prediction_loss�G�>

reg_loss�+w<


total_loss�>


accuracy_1)\?x�&�]       a[��	*�,_��A�+*O

prediction_loss�G�>

reg_loss�+w<


total_loss�>


accuracy_1)\?z��]       a[��	G8�,_��A�+*O

prediction_loss��?

reg_loss�+w<


total_lossIv?


accuracy_1���>$rQ�]       a[��	TW�,_��A�+*O

prediction_loss�z?

reg_loss�+w<


total_loss�W?


accuracy_1=
�>�k��]       a[��	�q�,_��A�+*O

prediction_loss��>

reg_loss�+w<


total_loss|>�>


accuracy_1q=
?�3�]       a[��	���,_��A�+*O

prediction_loss��>

reg_loss�+w<


total_loss|>�>


accuracy_1q=
?�z�]       a[��	%��,_��A�+*O

prediction_lossR�?

reg_loss�+w<


total_loss �"?


accuracy_1\��>>�']       a[��	:��,_��A�+*O

prediction_loss���>

reg_loss�+w<


total_loss�{�>


accuracy_1�?q��]       a[��	-�,_��A�+*O

prediction_loss)\?

reg_loss|+w<


total_loss�8?


accuracy_1�G�>Ww,]       a[��	$�,_��A�+*O

prediction_loss���>

reg_losss+w<


total_loss)��>


accuracy_1��?�.]       a[��	�=�,_��A�+*O

prediction_loss   ?

reg_lossi+w<


total_loss��?


accuracy_1   ?#���]       a[��	fj�,_��A�+*O

prediction_loss   ?

reg_loss^+w<


total_loss��?


accuracy_1   ?�IS]       a[��	���,_��A�+*O

prediction_loss�G�>

reg_lossU+w<


total_loss	�>


accuracy_1)\?��)"]       a[��	N��,_��A�+*O

prediction_loss�z?

reg_lossJ+w<


total_loss�W?


accuracy_1=
�>@i@]       a[��	���,_��A�+*O

prediction_loss��>

reg_loss@+w<


total_lossy>�>


accuracy_1q=
?��]       a[��	`V�,_��A�+*O

prediction_loss   ?

reg_loss6+w<


total_loss��?


accuracy_1   ?����]       a[��	>w�,_��A�+*O

prediction_lossR�?

reg_loss,+w<


total_loss��"?


accuracy_1\��>�� ]       a[��	ݖ�,_��A�+*O

prediction_loss)\?

reg_loss"+w<


total_loss�8?


accuracy_1�G�>�ٺ]       a[��	���,_��A�+*O

prediction_loss   ?

reg_loss+w<


total_loss��?


accuracy_1   ?���]       a[��	Y��,_��A�+*O

prediction_loss�Q�>

reg_loss+w<


total_lossD�>


accuracy_1
�#?��|�]       a[��	��,_��A�+*O

prediction_loss=
�>

reg_loss+w<


total_loss���>


accuracy_1�z?�ڛ	]       a[��	y �,_��A�+*O

prediction_loss
�#?

reg_loss�*w<


total_loss��'?


accuracy_1�Q�>Q�<f]       a[��	�A�,_��A�+*O

prediction_loss)\?

reg_loss�*w<


total_loss�8?


accuracy_1�G�>h�>,]       a[��	h�,_��A�+*O

prediction_loss)\?

reg_loss�*w<


total_loss�8?


accuracy_1�G�>U�?]       a[��	 ��,_��A�+*O

prediction_loss��>

reg_loss�*w<


total_lossv>�>


accuracy_1q=
?��G]       a[��	X��,_��A�+*O

prediction_loss�?

reg_loss�*w<


total_lossc�?


accuracy_1���>��\]       a[��	���,_��A�+*O

prediction_loss�z?

reg_loss�*w<


total_loss�W?


accuracy_1=
�>��{]       a[��	���,_��A�+*O

prediction_loss   ?

reg_loss�*w<


total_loss��?


accuracy_1   ?�+X�]       a[��	��,_��A�+*O

prediction_loss�G�>

reg_loss�*w<


total_loss�>


accuracy_1)\?����]       a[��	Y2�,_��A�+*O

prediction_loss���>

reg_loss�*w<


total_loss�{�>


accuracy_1�?��]       a[��	8N�,_��A�+*O

prediction_loss=
�>

reg_loss�*w<


total_loss���>


accuracy_1�z?6�'!]       a[��	Xr�,_��A�+*O

prediction_loss
�#?

reg_loss�*w<


total_loss��'?


accuracy_1�Q�>�^��]       a[��	ޒ�,_��A�+*O

prediction_lossq=
?

reg_loss�*w<


total_loss?


accuracy_1��>���T]       a[��	���,_��A�+*O

prediction_loss   ?

reg_lossy*w<


total_loss��?


accuracy_1   ?Z�[�]       a[��	:��,_��A�+*O

prediction_loss�G�>

reg_losso*w<


total_loss�>


accuracy_1)\?���]       a[��	t��,_��A�+*O

prediction_loss=
�>

reg_losse*w<


total_loss���>


accuracy_1�z?�UFc]       a[��	=�,_��A�+*O

prediction_loss��?

reg_loss]*w<


total_lossCv?


accuracy_1���>㞊�]       a[��	�+�,_��A�+*O

prediction_loss��?

reg_lossT*w<


total_lossCv?


accuracy_1���>�4]       a[��	�N�,_��A�+*O

prediction_loss   ?

reg_lossG*w<


total_loss��?


accuracy_1   ?gV%L]       a[��	�n�,_��A�+*O

prediction_loss���>

reg_loss=*w<


total_loss�{�>


accuracy_1�?��]       a[��	���,_��A�+*O

prediction_loss���>

reg_loss4*w<


total_loss�{�>


accuracy_1�?���]       a[��	/��,_��A�+*O

prediction_loss�?

reg_loss)*w<


total_lossa�?


accuracy_1���>l�gX]       a[��	���,_��A�+*O

prediction_loss�G�>

reg_loss*w<


total_loss� �>


accuracy_1)\?̾�]       a[��	���,_��A�+*O

prediction_loss�?

reg_loss*w<


total_loss`�?


accuracy_1���>0A�$]       a[��	���,_��A�+*O

prediction_loss
�#?

reg_loss	*w<


total_loss��'?


accuracy_1�Q�>���]       a[��	*�,_��A�+*O

prediction_loss\��>

reg_loss�)w<


total_loss�H�>


accuracy_1R�?M��]       a[��	�6�,_��A�+*O

prediction_loss   ?

reg_loss�)w<


total_loss��?


accuracy_1   ?�:�,]       a[��	�R�,_��A�+*O

prediction_loss\��>

reg_loss�)w<


total_loss�H�>


accuracy_1R�?I���]       a[��	�m�,_��A�+*O

prediction_lossq=
?

reg_loss�)w<


total_loss?


accuracy_1��>O��]       a[��	���,_��A�+*O

prediction_loss���>

reg_loss�)w<


total_loss�{�>


accuracy_1�?{M);]       a[��	���,_��A�+*O

prediction_loss)\?

reg_loss�)w<


total_loss�8?


accuracy_1�G�>��Ⱥ]       a[��	"��,_��A�+*O

prediction_loss�?

reg_loss�)w<


total_loss_�?


accuracy_1���>�|��]       a[��	*��,_��A�+*O

prediction_loss�?

reg_loss�)w<


total_loss_�?


accuracy_1���>H�}�]       a[��	��,_��A�+*O

prediction_loss�G�>

reg_loss�)w<


total_loss� �>


accuracy_1)\?�r�]       a[��	�#�,_��A�+*O

prediction_loss   ?

reg_loss�)w<


total_loss��?


accuracy_1   ?��C]       a[��	=E�,_��A�+*O

prediction_loss�?

reg_loss�)w<


total_loss^�?


accuracy_1���>�G]       a[��	ca�,_��A�+*O

prediction_loss   ?

reg_loss�)w<


total_loss��?


accuracy_1   ?�d��]       a[��	9}�,_��A�+*O

prediction_loss�G�>

reg_loss�)w<


total_loss� �>


accuracy_1)\?~Q'�]       a[��	O��,_��A�+*O

prediction_loss�G�>

reg_loss{)w<


total_loss� �>


accuracy_1)\?�DN�]       a[��	���,_��A�+*O

prediction_loss=
�>

reg_lossp)w<


total_loss���>


accuracy_1�z?�BQ]       a[��	���,_��A�+*O

prediction_loss��>

reg_lossf)w<


total_lossj>�>


accuracy_1q=
?ݧ��]       a[��	L��,_��A�+*O

prediction_loss   ?

reg_loss])w<


total_loss��?


accuracy_1   ?e ]       a[��	���,_��A�+*O

prediction_loss��>

reg_lossR)w<


total_lossj>�>


accuracy_1q=
?�ڡ�]       a[��	�# -_��A�+*O

prediction_loss��?

reg_lossH)w<


total_loss?v?


accuracy_1���>�H8�]       a[��	@ -_��A�+*O

prediction_loss��?

reg_loss>)w<


total_loss?v?


accuracy_1���>1��]       a[��	�] -_��A�+*O

prediction_lossR�?

reg_loss5)w<


total_loss��"?


accuracy_1\��>/��]       a[��	yx -_��A�+*O

prediction_loss�z?

reg_loss*)w<


total_loss�W?


accuracy_1=
�>�"\^]       a[��	B� -_��A�+*O

prediction_loss�z?

reg_loss!)w<


total_loss�W?


accuracy_1=
�>M�(\]       a[��	� -_��A�+*O

prediction_loss{�>

reg_loss)w<


total_loss�͵>


accuracy_1��(?�@��]       a[��	J� -_��A�+*O

prediction_loss���>

reg_loss)w<


total_loss�{�>


accuracy_1�?(\`�]       a[��	�� -_��A�+*O

prediction_lossq=
?

reg_loss)w<


total_loss?


accuracy_1��>h�9S]       a[��	�-_��A�+*O

prediction_loss�?

reg_loss�(w<


total_loss\�?


accuracy_1���>R1��]       a[��	�--_��A�+*O

prediction_loss�G�>

reg_loss�(w<


total_loss� �>


accuracy_1)\?,jg[]       a[��	�I-_��A�+*O

prediction_loss��>

reg_loss�(w<


total_lossf>�>


accuracy_1q=
?]f�\]       a[��	�b-_��A�+*O

prediction_loss��>

reg_loss�(w<


total_lossf>�>


accuracy_1q=
?��Qd]       a[��	�~-_��A�+*O

prediction_loss333?

reg_loss�(w<


total_loss�7?


accuracy_1���>f/�$]       a[��	Ǜ-_��A�+*O

prediction_loss���>

reg_loss�(w<


total_loss�{�>


accuracy_1�?E2��]       a[��	�-_��A�+*O

prediction_lossq=
?

reg_loss�(w<


total_loss?


accuracy_1��>��8�]       a[��	�-_��A�+*O

prediction_loss��>

reg_loss�(w<


total_losse>�>


accuracy_1q=
?���!]       a[��	��-_��A�+*O

prediction_loss�?

reg_loss�(w<


total_loss[�?


accuracy_1���>i�$]       a[��	�-_��A�+*O

prediction_loss��?

reg_loss�(w<


total_loss<v?


accuracy_1���>f0��]       a[��	!@-_��A�+*O

prediction_loss���>

reg_loss�(w<


total_loss�{�>


accuracy_1�?~Y]       a[��	)\-_��A�+*O

prediction_loss��?

reg_loss�(w<


total_loss<v?


accuracy_1���>�;�p]       a[��	Jy-_��A�+*O

prediction_loss���>

reg_loss~(w<


total_loss�R�>


accuracy_1333?�T1]       a[��	��-_��A�+*O

prediction_loss�G�>

reg_lossr(w<


total_loss� �>


accuracy_1)\?}
u�]       a[��	��-_��A�+*O

prediction_loss�z?

reg_lossg(w<


total_loss�W?


accuracy_1=
�>d]       a[��	��-_��A�+*O

prediction_loss\��>

reg_loss\(w<


total_loss�H�>


accuracy_1R�?���_]       a[��	4�-_��A�+*O

prediction_loss�z?

reg_lossQ(w<


total_loss�W?


accuracy_1=
�>(���]       a[��	^-_��A�+*O

prediction_lossq=
?

reg_lossH(w<


total_loss?


accuracy_1��>���]       a[��	Z.-_��A�+*O

prediction_lossq=
?

reg_loss;(w<


total_loss?


accuracy_1��>���]       a[��	�J-_��A�+*O

prediction_loss��>

reg_loss3(w<


total_lossa>�>


accuracy_1q=
?Jx]       a[��	gc-_��A�+*O

prediction_loss
�#?

reg_loss'(w<


total_loss��'?


accuracy_1�Q�>)�`�]       a[��	��-_��A�+*O

prediction_loss��?

reg_loss(w<


total_loss:v?


accuracy_1���>k���]       a[��	�-_��A�+*O

prediction_loss=
�>

reg_loss(w<


total_loss~��>


accuracy_1�z?ϱ�]       a[��	'�-_��A�+*O

prediction_loss   ?

reg_loss
(w<


total_loss��?


accuracy_1   ?��G]       a[��	��-_��A�+*O

prediction_lossq=
?

reg_loss�'w<


total_loss?


accuracy_1��>�6�]       a[��	`-_��A�+*O

prediction_loss�?

reg_loss�'w<


total_lossX�?


accuracy_1���>G�D*]       a[��	g|-_��A�+*O

prediction_loss   ?

reg_loss�'w<


total_loss��?


accuracy_1   ?�KCv]       a[��	��-_��A�+*O

prediction_loss��>

reg_loss�'w<


total_loss^>�>


accuracy_1q=
?A{�]       a[��	J�-_��A�+*O

prediction_loss�z?

reg_loss�'w<


total_loss�W?


accuracy_1=
�>��+]       a[��	,�-_��A�+*O

prediction_lossq=
?

reg_loss�'w<


total_loss?


accuracy_1��>f�w]       a[��	��-_��A�+*O

prediction_loss   ?

reg_loss�'w<


total_loss��?


accuracy_1   ?j�]       a[��	�-_��A�+*O

prediction_loss)\?

reg_loss�'w<


total_loss�8?


accuracy_1�G�>y��]       a[��	�.-_��A�+*O

prediction_loss   ?

reg_loss�'w<


total_loss��?


accuracy_1   ?DT��]       a[��	+N-_��A�+*O

prediction_loss���>

reg_loss�'w<


total_loss�{�>


accuracy_1�?���]       a[��	bg-_��A�+*O

prediction_loss��?

reg_loss�'w<


total_loss8v?


accuracy_1���>���]       a[��	��-_��A�+*O

prediction_loss)\?

reg_loss�'w<


total_loss�8?


accuracy_1�G�>�9W]       a[��	��-_��A�+*O

prediction_loss���>

reg_lossx'w<


total_loss	��>


accuracy_1��?#�'�]       a[��	�-_��A�+*O

prediction_loss   ?

reg_lossn'w<


total_loss��?


accuracy_1   ?��Q6]       a[��	��-_��A�+*O

prediction_loss��>

reg_lossd'w<


total_lossZ>�>


accuracy_1q=
?��w]       a[��	D�-_��A�+*O

prediction_loss)\?

reg_loss['w<


total_loss�8?


accuracy_1�G�>VT[H]       a[��	�-_��A�+*O

prediction_loss���>

reg_lossO'w<


total_loss�{�>


accuracy_1�?��@
]       a[��	�/-_��A�+*O

prediction_loss�G�>

reg_lossE'w<


total_loss� �>


accuracy_1)\?��}�]       a[��	�v-_��A�+*O

prediction_loss�z?

reg_loss9'w<


total_loss~W?


accuracy_1=
�>�?�\]       a[��	��-_��A�+*O

prediction_loss�z?

reg_loss/'w<


total_loss~W?


accuracy_1=
�>���o]       a[��	g�-_��A�+*O

prediction_loss=
�>

reg_loss$'w<


total_lossv��>


accuracy_1�z?K��]       a[��	�-_��A�+*O

prediction_loss�?

reg_loss'w<


total_lossT�?


accuracy_1���>ӺO�]       a[��	�\-_��A�+*O

prediction_loss��?

reg_loss'w<


total_loss6v?


accuracy_1���>��]       a[��	R|-_��A�+*O

prediction_loss�G�>

reg_loss'w<


total_loss� �>


accuracy_1)\?	���]       a[��	�-_��A�+*O

prediction_loss��>

reg_loss�&w<


total_lossW>�>


accuracy_1q=
?�9=]       a[��	G�-_��A�+*O

prediction_loss���>

reg_loss�&w<


total_loss�{�>


accuracy_1�?N���]       a[��	��-_��A�+*O

prediction_loss   ?

reg_loss�&w<


total_loss��?


accuracy_1   ?�n
 ]       a[��	�-_��A�+*O

prediction_loss
ף>

reg_loss�&w<


total_lossA��>


accuracy_1{.?��ˌ]       a[��	�:-_��A�+*O

prediction_loss�G�>

reg_loss�&w<


total_loss� �>


accuracy_1)\?"��n]       a[��	�`-_��A�+*O

prediction_loss=
�>

reg_loss�&w<


total_losss��>


accuracy_1�z?��A�]       a[��	-_��A�+*O

prediction_loss���>

reg_loss�&w<


total_loss�{�>


accuracy_1�?
��]       a[��	Ƥ-_��A�+*O

prediction_loss��?

reg_loss�&w<


total_loss5v?


accuracy_1���>��vJ]       a[��	��-_��A�,*O

prediction_loss���>

reg_loss�&w<


total_loss�{�>


accuracy_1�?ѫ�]       a[��	��-_��A�,*O

prediction_lossq=
?

reg_loss�&w<


total_loss?


accuracy_1��>�9��]       a[��		-_��A�,*O

prediction_loss���>

reg_loss�&w<


total_loss�{�>


accuracy_1�?�$T]       a[��	5	-_��A�,*O

prediction_loss)\?

reg_loss�&w<


total_loss�8?


accuracy_1�G�>4S|Y]       a[��	O[	-_��A�,*O

prediction_lossq=
?

reg_loss~&w<


total_loss?


accuracy_1��>��]       a[��	�v	-_��A�,*O

prediction_loss��>

reg_losst&w<


total_lossS>�>


accuracy_1q=
?���]       a[��	�	-_��A�,*O

prediction_loss���>

reg_lossj&w<


total_loss ��>


accuracy_1��?7�B�]       a[��	��	-_��A�,*O

prediction_loss��?

reg_lossa&w<


total_loss4v?


accuracy_1���>OkI�]       a[��	&�	-_��A�,*O

prediction_loss)\?

reg_lossV&w<


total_loss�8?


accuracy_1�G�>=�Ѿ]       a[��	�
-_��A�,*O

prediction_loss\��>

reg_lossL&w<


total_loss�H�>


accuracy_1R�?��F]       a[��	.<
-_��A�,*O

prediction_loss�G�>

reg_lossC&w<


total_loss� �>


accuracy_1)\?;ï]       a[��	-`
-_��A�,*O

prediction_loss��>

reg_loss9&w<


total_lossQ>�>


accuracy_1q=
?<�t3]       a[��	�}
-_��A�,*O

prediction_loss\��>

reg_loss.&w<


total_loss�H�>


accuracy_1R�?��&]       a[��	��
-_��A�,*O

prediction_loss��(?

reg_loss$&w<


total_loss\�,?


accuracy_1{�>x=�2]       a[��	��
-_��A�,*O

prediction_loss�G�>

reg_loss&w<


total_loss� �>


accuracy_1)\?��U�]       a[��	��
-_��A�,*O

prediction_loss��>

reg_loss&w<


total_lossO>�>


accuracy_1q=
?����]       a[��	�-_��A�,*O

prediction_loss��>

reg_loss&w<


total_lossO>�>


accuracy_1q=
?:�y]       a[��	�1-_��A�,*O

prediction_loss��>

reg_loss�%w<


total_lossO>�>


accuracy_1q=
?E�]       a[��	�Q-_��A�,*O

prediction_loss��>

reg_loss�%w<


total_lossO>�>


accuracy_1q=
?ziO�]       a[��	uv-_��A�,*O

prediction_loss���>

reg_loss�%w<


total_loss���>


accuracy_1��?m�E]       a[��	ɒ-_��A�,*O

prediction_loss)\?

reg_loss�%w<


total_loss�8?


accuracy_1�G�>���]       a[��	�-_��A�,*O

prediction_loss���>

reg_loss�%w<


total_loss���>


accuracy_1��?��]       a[��	g�-_��A�,*O

prediction_loss�z?

reg_loss�%w<


total_lossxW?


accuracy_1=
�>M��M]       a[��	-_��A�,*O

prediction_loss��>

reg_loss�%w<


total_lossM>�>


accuracy_1q=
?dS/�]       a[��	�0-_��A�,*O

prediction_loss)\?

reg_loss�%w<


total_loss�8?


accuracy_1�G�>�$�]       a[��	�O-_��A�,*O

prediction_loss���>

reg_loss�%w<


total_loss�{�>


accuracy_1�?{���]       a[��	Lm-_��A�,*O

prediction_loss=
�>

reg_loss�%w<


total_lossj��>


accuracy_1�z?�QJF]       a[��	�-_��A�,*O

prediction_loss��>

reg_loss�%w<


total_lossL>�>


accuracy_1q=
? ��]       a[��	\�-_��A�,*O

prediction_loss���>

reg_loss�%w<


total_loss���>


accuracy_1��?�v~*]       a[��	d�-_��A�,*O

prediction_loss�?

reg_loss%w<


total_lossN�?


accuracy_1���>��>X]       a[��	��-_��A�,*O

prediction_loss=
�>

reg_lossu%w<


total_lossi��>


accuracy_1�z?A֏�]       a[��		-_��A�,*O

prediction_loss�G�>

reg_lossj%w<


total_loss� �>


accuracy_1)\?]�5�]       a[��	"-_��A�,*O

prediction_loss   ?

reg_loss`%w<


total_loss��?


accuracy_1   ?N�@�]       a[��	�?-_��A�,*O

prediction_loss��?

reg_lossU%w<


total_loss/v?


accuracy_1���>Pc�]       a[��	�\-_��A�,*O

prediction_lossq=
?

reg_lossL%w<


total_loss?


accuracy_1��>�he]       a[��	 z-_��A�,*O

prediction_loss=
�>

reg_lossA%w<


total_lossg��>


accuracy_1�z?�<��]       a[��	h�-_��A�,*O

prediction_loss�G�>

reg_loss5%w<


total_loss� �>


accuracy_1)\?�D�]       a[��	��-_��A�,*O

prediction_loss��?

reg_loss,%w<


total_loss/v?


accuracy_1���>ՍG]       a[��	N�-_��A�,*O

prediction_loss��(?

reg_loss!%w<


total_lossX�,?


accuracy_1{�>�$(�]       a[��	c�-_��A�,*O

prediction_loss�G�>

reg_loss%w<


total_loss� �>


accuracy_1)\?� �]       a[��	-_��A�,*O

prediction_loss���>

reg_loss%w<


total_loss�{�>


accuracy_1�?L)µ]       a[��	p"-_��A�,*O

prediction_loss��>

reg_loss%w<


total_lossG>�>


accuracy_1q=
?(I2,]       a[��	q;-_��A�,*O

prediction_loss�z?

reg_loss�$w<


total_lossuW?


accuracy_1=
�>_RP]       a[��	�R-_��A�,*O

prediction_loss)\?

reg_loss�$w<


total_loss�8?


accuracy_1�G�>+4�(]       a[��	rm-_��A�,*O

prediction_loss=
�>

reg_loss�$w<


total_lossd��>


accuracy_1�z?/M]       a[��	ۋ-_��A�,*O

prediction_loss
�#?

reg_loss�$w<


total_loss��'?


accuracy_1�Q�>;���]       a[��	�-_��A�,*O

prediction_loss   ?

reg_loss�$w<


total_loss��?


accuracy_1   ?Q,d�]       a[��	��-_��A�,*O

prediction_loss   ?

reg_loss�$w<


total_loss��?


accuracy_1   ?؂�]       a[��	]�-_��A�,*O

prediction_lossR�?

reg_loss�$w<


total_loss�"?


accuracy_1\��>��6p]       a[��	k�-_��A�,*O

prediction_loss�G�>

reg_loss�$w<


total_loss� �>


accuracy_1)\?���~]       a[��	�	-_��A�,*O

prediction_loss   ?

reg_loss�$w<


total_loss��?


accuracy_1   ?��]]       a[��	1'-_��A�,*O

prediction_loss�?

reg_loss�$w<


total_lossJ�?


accuracy_1���>wD�/]       a[��	RC-_��A�,*O

prediction_loss�G�>

reg_loss�$w<


total_loss� �>


accuracy_1)\?'(9�]       a[��	a-_��A�,*O

prediction_loss�G�>

reg_loss�$w<


total_loss� �>


accuracy_1)\?=���]       a[��	X�-_��A�,*O

prediction_loss=
�>

reg_loss�$w<


total_lossa��>


accuracy_1�z?�R�<]       a[��	 �-_��A�,*O

prediction_loss�?

reg_lossv$w<


total_lossJ�?


accuracy_1���>��]       a[��	�-_��A�,*O

prediction_loss�G�>

reg_lossk$w<


total_loss� �>


accuracy_1)\?�>u]       a[��	e�-_��A�,*O

prediction_loss�?

reg_loss`$w<


total_lossJ�?


accuracy_1���> eV6]       a[��	�-_��A�,*O

prediction_loss���>

reg_lossV$w<


total_loss�{�>


accuracy_1�?k�c�]       a[��	�-_��A�,*O

prediction_lossR�?

reg_lossJ$w<


total_loss�"?


accuracy_1\��>q �]       a[��	s--_��A�,*O

prediction_loss�?

reg_loss@$w<


total_lossI�?


accuracy_1���>�eU]       a[��	�K-_��A�,*O

prediction_loss   ?

reg_loss7$w<


total_loss��?


accuracy_1   ?�xL�]       a[��	an-_��A�,*O

prediction_loss�G�>

reg_loss*$w<


total_loss� �>


accuracy_1)\?���]       a[��	;�-_��A�,*O

prediction_loss�G�>

reg_loss"$w<


total_loss� �>


accuracy_1)\?Z,4M]       a[��	m�-_��A�,*O

prediction_loss   ?

reg_loss$w<


total_loss��?


accuracy_1   ?�uf]       a[��	��-_��A�,*O

prediction_loss��>

reg_loss$w<


total_loss?>�>


accuracy_1q=
?�1W]       a[��	��-_��A�,*O

prediction_loss���>

reg_loss$w<


total_loss�{�>


accuracy_1�?��}�]       a[��	7�-_��A�,*O

prediction_loss�z?

reg_loss�#w<


total_lossqW?


accuracy_1=
�>���B]       a[��	f-_��A�,*O

prediction_loss   ?

reg_loss�#w<


total_loss��?


accuracy_1   ?��M]       a[��	se-_��A�,*O

prediction_loss{�>

reg_loss�#w<


total_loss�͵>


accuracy_1��(?�u��]       a[��	A�-_��A�,*O

prediction_loss�G�>

reg_loss�#w<


total_loss� �>


accuracy_1)\?�:��]       a[��	�-_��A�,*O

prediction_loss���>

reg_loss�#w<


total_loss�{�>


accuracy_1�?�PȊ]       a[��	��-_��A�,*O

prediction_loss��>

reg_loss�#w<


total_loss=>�>


accuracy_1q=
?�d��]       a[��	O�-_��A�,*O

prediction_loss)\?

reg_loss�#w<


total_loss�8?


accuracy_1�G�>�f�d]       a[��	 �-_��A�,*O

prediction_loss�z?

reg_loss�#w<


total_losspW?


accuracy_1=
�>5\��]       a[��	i�-_��A�,*O

prediction_loss��>

reg_loss�#w<


total_loss<>�>


accuracy_1q=
?��$ ]       a[��	�-_��A�,*O

prediction_loss   ?

reg_loss�#w<


total_loss��?


accuracy_1   ?JW�]       a[��	{2-_��A�,*O

prediction_loss�G�>

reg_loss�#w<


total_loss� �>


accuracy_1)\?�'n�]       a[��	L-_��A�,*O

prediction_loss�z?

reg_loss|#w<


total_lossoW?


accuracy_1=
�>7U07]       a[��	cd-_��A�,*O

prediction_loss��?

reg_lossr#w<


total_loss(v?


accuracy_1���>A!�n]       a[��	9~-_��A�,*O

prediction_loss��>

reg_lossg#w<


total_loss:>�>


accuracy_1q=
?�j�]       a[��	-�-_��A�,*O

prediction_loss��>

reg_loss]#w<


total_loss:>�>


accuracy_1q=
?\���]       a[��	�-_��A�,*O

prediction_loss   ?

reg_lossS#w<


total_loss��?


accuracy_1   ?�_�o]       a[��	��-_��A�,*O

prediction_loss��>

reg_lossJ#w<


total_loss9>�>


accuracy_1q=
?Yv"�]       a[��	�-_��A�,*O

prediction_lossq=
?

reg_loss?#w<


total_loss�?


accuracy_1��>jR]       a[��	�-_��A�,*O

prediction_lossq=
?

reg_loss6#w<


total_loss�?


accuracy_1��>ŘX9]       a[��	�-_��A�,*O

prediction_lossq=
?

reg_loss+#w<


total_loss�?


accuracy_1��>�|��]       a[��	�8-_��A�,*O

prediction_loss
�#?

reg_loss"#w<


total_loss��'?


accuracy_1�Q�>�%Z�]       a[��	}u-_��A�,*O

prediction_loss�z?

reg_loss#w<


total_lossmW?


accuracy_1=
�>Dѡ�]       a[��	��-_��A�,*O

prediction_lossq=
?

reg_loss#w<


total_loss�?


accuracy_1��>�ES�]       a[��	��-_��A�,*O

prediction_loss��>

reg_loss#w<


total_loss7>�>


accuracy_1q=
?-��]       a[��	u�-_��A�,*O

prediction_loss�z?

reg_loss�"w<


total_lossmW?


accuracy_1=
�>�m��]       a[��	o)-_��A�,*O

prediction_loss���>

reg_loss�"w<


total_loss�{�>


accuracy_1�?�Ά�]       a[��	I-_��A�,*O

prediction_loss\��>

reg_loss�"w<


total_losssH�>


accuracy_1R�?�=�]       a[��	�g-_��A�,*O

prediction_loss)\?

reg_loss�"w<


total_loss�8?


accuracy_1�G�>LuGy]       a[��	U�-_��A�,*O

prediction_loss���>

reg_loss�"w<


total_loss��>


accuracy_1��?�iz�]       a[��	Ϥ-_��A�,*O

prediction_loss   ?

reg_loss�"w<


total_loss��?


accuracy_1   ?�;Y+]       a[��	��-_��A�,*O

prediction_loss{.?

reg_loss�"w<


total_loss�1?


accuracy_1
ף>t׼]       a[��	��-_��A�,*O

prediction_loss�G�>

reg_loss�"w<


total_loss� �>


accuracy_1)\?���]       a[��	��-_��A�,*O

prediction_loss�?

reg_loss�"w<


total_lossC�?


accuracy_1���>���d]       a[��	-_��A�,*O

prediction_lossq=
?

reg_loss�"w<


total_loss�?


accuracy_1��>��2{]       a[��	�3-_��A�,*O

prediction_loss�G�>

reg_loss�"w<


total_loss� �>


accuracy_1)\?��~�]       a[��	�J-_��A�,*O

prediction_loss��>

reg_loss�"w<


total_loss3>�>


accuracy_1q=
?�x�]       a[��	,d-_��A�,*O

prediction_lossq=
?

reg_loss{"w<


total_loss�?


accuracy_1��>, π]       a[��	�|-_��A�,*O

prediction_loss��>

reg_lossr"w<


total_loss3>�>


accuracy_1q=
?�;$]       a[��	��-_��A�,*O

prediction_loss�G�>

reg_losse"w<


total_loss� �>


accuracy_1)\?��O=]       a[��	�-_��A�,*O

prediction_lossq=
?

reg_loss["w<


total_loss�?


accuracy_1��>�M�]       a[��	��-_��A�,*O

prediction_loss�?

reg_lossQ"w<


total_lossA�?


accuracy_1���>&�+]       a[��	��-_��A�,*O

prediction_lossR�?

reg_lossG"w<


total_loss۔"?


accuracy_1\��>���J]       a[��	��-_��A�,*O

prediction_loss=
�>

reg_loss="w<


total_lossO��>


accuracy_1�z?[r=�]       a[��	e-_��A�,*O

prediction_loss�G�>

reg_loss2"w<


total_loss� �>


accuracy_1)\?��2?]       a[��	f1-_��A�,*O

prediction_lossq=
?

reg_loss)"w<


total_loss�?


accuracy_1��>;�{y]       a[��	�H-_��A�,*O

prediction_loss��?

reg_loss"w<


total_loss"v?


accuracy_1���>�eQ]       a[��	�c-_��A�,*O

prediction_loss�G�>

reg_loss"w<


total_loss� �>


accuracy_1)\?nGx�]       a[��	�}-_��A�,*O

prediction_loss   ?

reg_loss"w<


total_loss��?


accuracy_1   ?��K3]       a[��	g�-_��A�,*O

prediction_lossq=
?

reg_loss�!w<


total_loss�?


accuracy_1��>
�f]       a[��	�-_��A�,*O

prediction_loss=
�>

reg_loss�!w<


total_lossM��>


accuracy_1�z?z�J]       a[��	k�-_��A�,*O

prediction_lossR�?

reg_loss�!w<


total_lossڔ"?


accuracy_1\��>���]       a[��	�-_��A�,*O

prediction_loss���>

reg_loss�!w<


total_loss�{�>


accuracy_1�?V9�a]       a[��	z9-_��A�,*O

prediction_loss   ?

reg_loss�!w<


total_loss��?


accuracy_1   ?=�K]       a[��	�^-_��A�,*O

prediction_loss=
�>

reg_loss�!w<


total_lossK��>


accuracy_1�z?�`dP]       a[��	�-_��A�,*O

prediction_loss)\?

reg_loss�!w<


total_loss�8?


accuracy_1�G�>���]       a[��	��-_��A�,*O

prediction_loss���>

reg_loss�!w<


total_lossۅ�>


accuracy_1��?�[�]       a[��	��-_��A�,*O

prediction_loss   ?

reg_loss�!w<


total_loss��?


accuracy_1   ?�|]       a[��	�-_��A�,*O

prediction_loss{�>

reg_loss�!w<


total_loss�͵>


accuracy_1��(?���~]       a[��	�%-_��A�,*O

prediction_loss=
�>

reg_loss�!w<


total_lossJ��>


accuracy_1�z?<�C)]       a[��	�L-_��A�,*O

prediction_loss��>

reg_loss�!w<


total_loss+>�>


accuracy_1q=
?�y��]       a[��	l{-_��A�,*O

prediction_loss=
�>

reg_loss�!w<


total_lossI��>


accuracy_1�z?�Af]       a[��	#�-_��A�-*O

prediction_loss)\?

reg_lossy!w<


total_loss�8?


accuracy_1�G�>�=c]       a[��	��-_��A�-*O

prediction_loss���>

reg_lossp!w<


total_loss�{�>


accuracy_1�?ks�j]       a[��	��-_��A�-*O

prediction_loss�z?

reg_lossc!w<


total_lossgW?


accuracy_1=
�>48(Z]       a[��	�-_��A�-*O

prediction_loss���>

reg_loss[!w<


total_loss�{�>


accuracy_1�?C�]       a[��	3-_��A�-*O

prediction_loss\��>

reg_lossR!w<


total_lossgH�>


accuracy_1R�?z�Bi]       a[��	�^-_��A�-*O

prediction_loss�?

reg_lossF!w<


total_loss=�?


accuracy_1���>]�0�]       a[��	��-_��A�-*O

prediction_loss��>

reg_loss<!w<


total_loss)>�>


accuracy_1q=
?��>�]       a[��	��-_��A�-*O

prediction_loss���>

reg_loss2!w<


total_loss�{�>


accuracy_1�?����]       a[��	��-_��A�-*O

prediction_loss���>

reg_loss'!w<


total_loss�{�>


accuracy_1�?�G�W]       a[��	�-_��A�-*O

prediction_loss�z?

reg_loss!w<


total_losseW?


accuracy_1=
�>`3]       a[��	�:-_��A�-*O

prediction_loss   ?

reg_loss!w<


total_loss��?


accuracy_1   ?[dhe]       a[��	�y-_��A�-*O

prediction_loss��>

reg_loss
!w<


total_loss'>�>


accuracy_1q=
?ߢ/M]       a[��	B�-_��A�-*O

prediction_loss�?

reg_loss !w<


total_loss<�?


accuracy_1���>(,��]       a[��	9�-_��A�-*O

prediction_loss��?

reg_loss� w<


total_lossv?


accuracy_1���>����]       a[��	��-_��A�-*O

prediction_loss���>

reg_loss� w<


total_loss�{�>


accuracy_1�?B7��]       a[��	j�-_��A�-*O

prediction_lossq=
?

reg_loss� w<


total_loss�?


accuracy_1��>�R�B]       a[��	T-_��A�-*O

prediction_lossq=
?

reg_loss� w<


total_loss�?


accuracy_1��>�:UB]       a[��	WC-_��A�-*O

prediction_loss   ?

reg_loss� w<


total_loss��?


accuracy_1   ?��,]       a[��	�g-_��A�-*O

prediction_loss�Q�>

reg_loss� w<


total_loss�
�>


accuracy_1
�#?��u]       a[��	ɐ-_��A�-*O

prediction_loss�?

reg_loss� w<


total_loss;�?


accuracy_1���>W�]       a[��	��-_��A�-*O

prediction_lossq=
?

reg_loss� w<


total_loss�?


accuracy_1��>X��]       a[��	-_��A�-*O

prediction_loss\��>

reg_loss� w<


total_lossaH�>


accuracy_1R�?8-s(]       a[��	�K-_��A�-*O

prediction_loss{.?

reg_loss� w<


total_loss��1?


accuracy_1
ף>L}�]       a[��	�p-_��A�-*O

prediction_loss���>

reg_loss� w<


total_loss�{�>


accuracy_1�?��-l]       a[��	$�-_��A�-*O

prediction_loss�z?

reg_loss w<


total_losscW?


accuracy_1=
�>�L]       a[��	P�-_��A�-*O

prediction_loss��>

reg_lossw w<


total_loss#>�>


accuracy_1q=
?��[]       a[��	d�-_��A�-*O

prediction_lossq=
?

reg_lossm w<


total_loss�?


accuracy_1��>�ځ]       a[��	�-_��A�-*O

prediction_loss���>

reg_lossd w<


total_loss�{�>


accuracy_1�?5p��]       a[��	�7-_��A�-*O

prediction_loss)\?

reg_lossX w<


total_loss�8?


accuracy_1�G�>�nbA]       a[��	{g-_��A�-*O

prediction_lossq=
?

reg_lossM w<


total_loss�?


accuracy_1��>֣�q]       a[��	/�-_��A�-*O

prediction_loss���>

reg_lossB w<


total_loss�R�>


accuracy_1333?H>]       a[��	��-_��A�-*O

prediction_loss�?

reg_loss9 w<


total_loss9�?


accuracy_1���>2�Z}]       a[��	/�-_��A�-*O

prediction_loss��>

reg_loss/ w<


total_loss >�>


accuracy_1q=
?x"�a]       a[��	�-_��A�-*O

prediction_loss�?

reg_loss% w<


total_loss9�?


accuracy_1���>^n�:]       a[��	a7-_��A�-*O

prediction_loss��?

reg_loss w<


total_lossv?


accuracy_1���>���]       a[��	�T-_��A�-*O

prediction_loss�?

reg_loss w<


total_loss8�?


accuracy_1���>���X]       a[��	�{-_��A�-*O

prediction_loss\��>

reg_loss w<


total_loss\H�>


accuracy_1R�?O[��]       a[��	˝-_��A�-*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>��i]       a[��	��-_��A�-*O

prediction_loss=
�>

reg_loss�w<


total_loss<��>


accuracy_1�z?��Κ]       a[��	j�-_��A�-*O

prediction_loss\��>

reg_loss�w<


total_loss[H�>


accuracy_1R�?��X�]       a[��	��-_��A�-*O

prediction_loss)\?

reg_loss�w<


total_loss�8?


accuracy_1�G�>��{�]       a[��	Va-_��A�-*O

prediction_loss�z?

reg_loss�w<


total_loss`W?


accuracy_1=
�>Q<pu]       a[��	A~-_��A�-*O

prediction_loss�?

reg_loss�w<


total_loss7�?


accuracy_1���>G ��]       a[��	
�-_��A�-*O

prediction_loss�z?

reg_loss�w<


total_loss`W?


accuracy_1=
�>0٘]       a[��	A�-_��A�-*O

prediction_loss=
�>

reg_loss�w<


total_loss:��>


accuracy_1�z?�N�]       a[��	��-_��A�-*O

prediction_loss��?

reg_loss�w<


total_lossv?


accuracy_1���>M��]       a[��	��-_��A�-*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>��Jf]       a[��	G -_��A�-*O

prediction_loss���>

reg_loss�w<


total_loss�{�>


accuracy_1�?�*(�]       a[��	;7 -_��A�-*O

prediction_loss)\?

reg_loss�w<


total_loss�8?


accuracy_1�G�>�V]       a[��	�Q -_��A�-*O

prediction_loss��>

reg_lossvw<


total_loss>�>


accuracy_1q=
?��"]       a[��	�l -_��A�-*O

prediction_loss���>

reg_losslw<


total_loss�{�>


accuracy_1�?z�]       a[��	�� -_��A�-*O

prediction_loss�z?

reg_lossbw<


total_loss_W?


accuracy_1=
�>a�_�]       a[��	� -_��A�-*O

prediction_loss�z?

reg_lossWw<


total_loss^W?


accuracy_1=
�>�#n]       a[��	�� -_��A�-*O

prediction_loss�Q�>

reg_lossNw<


total_loss�
�>


accuracy_1
�#?v�B�]       a[��	�� -_��A�-*O

prediction_loss�?

reg_lossBw<


total_loss5�?


accuracy_1���>�#�]       a[��	D� -_��A�-*O

prediction_loss   ?

reg_loss9w<


total_loss}�?


accuracy_1   ?'|�k]       a[��	!-_��A�-*O

prediction_loss=
�>

reg_loss.w<


total_loss6��>


accuracy_1�z?p'�+]       a[��	>A!-_��A�-*O

prediction_loss)\?

reg_loss%w<


total_loss�8?


accuracy_1�G�>(�]       a[��	�o!-_��A�-*O

prediction_loss�?

reg_lossw<


total_loss4�?


accuracy_1���>bV�]       a[��	�!-_��A�-*O

prediction_lossq=
?

reg_lossw<


total_loss�?


accuracy_1��>���]       a[��	��!-_��A�-*O

prediction_loss���>

reg_lossw<


total_lossŅ�>


accuracy_1��?,+.]       a[��	��!-_��A�-*O

prediction_loss���>

reg_loss�w<


total_loss�{�>


accuracy_1�?;��A]       a[��	R"-_��A�-*O

prediction_loss=
�>

reg_loss�w<


total_loss4��>


accuracy_1�z?��Y]       a[��	;�"-_��A�-*O

prediction_loss)\?

reg_loss�w<


total_loss�8?


accuracy_1�G�>�~h?]       a[��	h�"-_��A�-*O

prediction_loss=
�>

reg_loss�w<


total_loss4��>


accuracy_1�z?�L�]       a[��	V�"-_��A�-*O

prediction_loss���>

reg_loss�w<


total_lossÅ�>


accuracy_1��?�`�]       a[��	� #-_��A�-*O

prediction_loss��?

reg_loss�w<


total_lossv?


accuracy_1���>��ߩ]       a[��	,*#-_��A�-*O

prediction_loss�?

reg_loss�w<


total_loss3�?


accuracy_1���>Cڌ�]       a[��	�J#-_��A�-*O

prediction_loss
�#?

reg_loss�w<


total_loss��'?


accuracy_1�Q�>?X�h]       a[��	jh#-_��A�-*O

prediction_loss���>

reg_loss�w<


total_loss�{�>


accuracy_1�?�;�h]       a[��	��#-_��A�-*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>e�X+]       a[��	7�#-_��A�-*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>���]       a[��	��#-_��A�-*O

prediction_loss�?

reg_loss�w<


total_loss2�?


accuracy_1���>I��r]       a[��	�#-_��A�-*O

prediction_lossq=
?

reg_loss|w<


total_loss�?


accuracy_1��>��p]       a[��	W$-_��A�-*O

prediction_loss���>

reg_lossrw<


total_loss�{�>


accuracy_1�?�|�]       a[��	�"$-_��A�-*O

prediction_loss���>

reg_lossgw<


total_loss�{�>


accuracy_1�?���E]       a[��	�A$-_��A�-*O

prediction_loss�?

reg_loss]w<


total_loss1�?


accuracy_1���>��f�]       a[��	�Z$-_��A�-*O

prediction_lossq=
?

reg_lossQw<


total_loss�?


accuracy_1��>���]       a[��	�p$-_��A�-*O

prediction_loss=
�>

reg_lossHw<


total_loss/��>


accuracy_1�z?��]       a[��	~�$-_��A�-*O

prediction_loss   ?

reg_loss=w<


total_lossy�?


accuracy_1   ?�I�L]       a[��	�$-_��A�-*O

prediction_loss�?

reg_loss3w<


total_loss1�?


accuracy_1���>�K]       a[��	��$-_��A�-*O

prediction_loss   ?

reg_loss*w<


total_lossy�?


accuracy_1   ?�+f�]       a[��	'�$-_��A�-*O

prediction_loss   ?

reg_loss w<


total_lossx�?


accuracy_1   ?��+q]       a[��	Q�$-_��A�-*O

prediction_loss�?

reg_lossw<


total_loss0�?


accuracy_1���>-�r]       a[��	%-_��A�-*O

prediction_loss��>

reg_loss
w<


total_loss>�>


accuracy_1q=
?�%�]       a[��	�4%-_��A�-*O

prediction_loss���>

reg_loss w<


total_loss{�>


accuracy_1�?�M��]       a[��	�M%-_��A�-*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1��?ov��]       a[��	j%-_��A�-*O

prediction_loss��(?

reg_loss�w<


total_loss;�,?


accuracy_1{�>q.HI]       a[��	�%-_��A�-*O

prediction_loss���>

reg_loss�w<


total_loss~{�>


accuracy_1�?�z5�]       a[��	�%-_��A�-*O

prediction_loss   ?

reg_loss�w<


total_lossw�?


accuracy_1   ?4�x�]       a[��	s�%-_��A�-*O

prediction_loss���>

reg_loss�w<


total_loss}{�>


accuracy_1�?��b"]       a[��	�%-_��A�-*O

prediction_loss��(?

reg_loss�w<


total_loss:�,?


accuracy_1{�>�6�U]       a[��	3�%-_��A�-*O

prediction_loss=
�>

reg_loss�w<


total_loss+��>


accuracy_1�z?�hG]       a[��	\&-_��A�-*O

prediction_loss=
�>

reg_loss�w<


total_loss*��>


accuracy_1�z?9�7]       a[��	O;&-_��A�-*O

prediction_loss��>

reg_loss�w<


total_loss>�>


accuracy_1q=
?S
q]       a[��	}Z&-_��A�-*O

prediction_loss�?

reg_loss�w<


total_loss.�?


accuracy_1���>o��Z]       a[��	Wy&-_��A�-*O

prediction_loss���>

reg_loss�w<


total_loss{{�>


accuracy_1�?�l��]       a[��	F�&-_��A�-*O

prediction_loss�?

reg_loss�w<


total_loss.�?


accuracy_1���>���]       a[��	��&-_��A�-*O

prediction_loss���>

reg_loss{w<


total_loss{{�>


accuracy_1�?�_�]       a[��	�&-_��A�-*O

prediction_loss�?

reg_lossow<


total_loss.�?


accuracy_1���>�t$]       a[��	��&-_��A�-*O

prediction_loss�?

reg_lossew<


total_loss.�?


accuracy_1���>�:x�]       a[��	'-_��A�-*O

prediction_loss���>

reg_loss\w<


total_lossz{�>


accuracy_1�?�!�C]       a[��	�1'-_��A�-*O

prediction_loss�?

reg_lossQw<


total_loss-�?


accuracy_1���>el?]       a[��	�N'-_��A�-*O

prediction_loss=
�>

reg_lossFw<


total_loss'��>


accuracy_1�z?·tH]       a[��	i'-_��A�-*O

prediction_loss=
�>

reg_loss:w<


total_loss'��>


accuracy_1�z?�&S]       a[��	��'-_��A�-*O

prediction_lossq=
?

reg_loss1w<


total_loss�?


accuracy_1��>%��%]       a[��	��'-_��A�-*O

prediction_loss��>

reg_loss&w<


total_loss>�>


accuracy_1q=
?�0}]       a[��	��'-_��A�-*O

prediction_loss���>

reg_lossw<


total_lossx{�>


accuracy_1�?S��]       a[��	��'-_��A�-*O

prediction_loss�z?

reg_lossw<


total_lossUW?


accuracy_1=
�>�t��]       a[��	}(-_��A�-*O

prediction_loss)\?

reg_lossw<


total_loss�8?


accuracy_1�G�>n��]       a[��	N*(-_��A�-*O

prediction_loss�z?

reg_loss�w<


total_lossUW?


accuracy_1=
�>띸T]       a[��	�H(-_��A�-*O

prediction_loss��(?

reg_loss�w<


total_loss7�,?


accuracy_1{�>9�]]       a[��	�h(-_��A�-*O

prediction_loss�?

reg_loss�w<


total_loss,�?


accuracy_1���>a�6]       a[��	�(-_��A�-*O

prediction_loss�?

reg_loss�w<


total_loss+�?


accuracy_1���>.O]       a[��	q�(-_��A�-*O

prediction_loss   ?

reg_loss�w<


total_losss�?


accuracy_1   ?��;]       a[��	_�(-_��A�-*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>���,]       a[��	��(-_��A�-*O

prediction_loss���>

reg_loss�w<


total_lossu{�>


accuracy_1�?�@
]       a[��	1)-_��A�-*O

prediction_loss=
�>

reg_loss�w<


total_loss#��>


accuracy_1�z?3ږ]       a[��	�')-_��A�-*O

prediction_loss=
�>

reg_loss�w<


total_loss"��>


accuracy_1�z?�Y~]       a[��	�M)-_��A�-*O

prediction_loss=
�>

reg_loss�w<


total_loss"��>


accuracy_1�z?��J]       a[��	vm)-_��A�-*O

prediction_loss�z?

reg_loss�w<


total_lossSW?


accuracy_1=
�>u�P�]       a[��	ŏ)-_��A�-*O

prediction_loss��>

reg_loss�w<


total_loss>�>


accuracy_1q=
?���]       a[��	��)-_��A�-*O

prediction_loss�z?

reg_loss�w<


total_lossSW?


accuracy_1=
�>����]       a[��	-�)-_��A�-*O

prediction_loss)\?

reg_lossyw<


total_loss�8?


accuracy_1�G�>�a!�]       a[��	x�)-_��A�-*O

prediction_loss���>

reg_lossow<


total_loss���>


accuracy_1��?�
�]       a[��	�*-_��A�-*O

prediction_loss�?

reg_lossew<


total_loss*�?


accuracy_1���>�}4!]       a[��	&9*-_��A�-*O

prediction_loss�?

reg_lossYw<


total_loss)�?


accuracy_1���>AbO�]       a[��	[[*-_��A�-*O

prediction_loss�G�>

reg_lossNw<


total_loss� �>


accuracy_1)\?J6��]       a[��	�{*-_��A�.*O

prediction_loss=
�>

reg_lossDw<


total_loss��>


accuracy_1�z?)�-]       a[��	Ϟ*-_��A�.*O

prediction_loss��>

reg_loss:w<


total_loss>�>


accuracy_1q=
?!,]       a[��	*�*-_��A�.*O

prediction_loss���>

reg_loss0w<


total_lossp{�>


accuracy_1�?���]       a[��	��*-_��A�.*O

prediction_loss)\?

reg_loss%w<


total_loss�8?


accuracy_1�G�>q��']       a[��	)+-_��A�.*O

prediction_loss=
�>

reg_lossw<


total_loss��>


accuracy_1�z?)�-S]       a[��	�'+-_��A�.*O

prediction_loss��>

reg_lossw<


total_loss >�>


accuracy_1q=
?�t�]       a[��	8M+-_��A�.*O

prediction_loss���>

reg_lossw<


total_losso{�>


accuracy_1�?Ux:]       a[��	�k+-_��A�.*O

prediction_loss��>

reg_loss�w<


total_loss�=�>


accuracy_1q=
?,�g�]       a[��	e�+-_��A�.*O

prediction_loss   ?

reg_loss�w<


total_lossp�?


accuracy_1   ?��a^]       a[��	��+-_��A�.*O

prediction_loss��(?

reg_loss�w<


total_loss3�,?


accuracy_1{�>Xh�]       a[��	��+-_��A�.*O

prediction_loss���>

reg_loss�w<


total_lossn{�>


accuracy_1�?�}.]       a[��	��+-_��A�.*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>P9��]       a[��	�,-_��A�.*O

prediction_loss�G�>

reg_loss�w<


total_loss� �>


accuracy_1)\?;/�]       a[��	��,-_��A�.*O

prediction_loss���>

reg_loss�w<


total_lossm{�>


accuracy_1�?X���]       a[��	+�,-_��A�.*O

prediction_loss)\?

reg_loss�w<


total_loss�8?


accuracy_1�G�>�YMz]       a[��	r�,-_��A�.*O

prediction_loss��>

reg_loss�w<


total_loss�=�>


accuracy_1q=
?�=�]       a[��	��,-_��A�.*O

prediction_loss
�#?

reg_loss�w<


total_lossx�'?


accuracy_1�Q�>�:}]       a[��	W--_��A�.*O

prediction_loss   ?

reg_loss�w<


total_lossn�?


accuracy_1   ?��5P]       a[��	>&--_��A�.*O

prediction_loss�Q�>

reg_loss�w<


total_loss�
�>


accuracy_1
�#?@:*]       a[��	,J--_��A�.*O

prediction_loss��>

reg_losszw<


total_loss�=�>


accuracy_1q=
?ߩ�]       a[��	�p--_��A�.*O

prediction_loss   ?

reg_lossnw<


total_lossn�?


accuracy_1   ?�(j^]       a[��	��--_��A�.*O

prediction_loss�z?

reg_lossew<


total_lossOW?


accuracy_1=
�>��K]       a[��	ٰ--_��A�.*O

prediction_loss\��>

reg_lossZw<


total_loss7H�>


accuracy_1R�?�]       a[��	��--_��A�.*O

prediction_loss��>

reg_lossQw<


total_loss�=�>


accuracy_1q=
?��x�]       a[��	�--_��A�.*O

prediction_loss�Q�>

reg_lossGw<


total_loss�
�>


accuracy_1
�#?�%�]       a[��	�.-_��A�.*O

prediction_loss�G�>

reg_loss;w<


total_loss� �>


accuracy_1)\?��M�]       a[��	�+.-_��A�.*O

prediction_loss�G�>

reg_loss1w<


total_loss� �>


accuracy_1)\?ٴlk]       a[��	�N.-_��A�.*O

prediction_loss�?

reg_loss'w<


total_loss%�?


accuracy_1���>(xs*]       a[��	#�.-_��A�.*O

prediction_loss���>

reg_lossw<


total_lossh{�>


accuracy_1�?[��Z]       a[��	S�.-_��A�.*O

prediction_loss   ?

reg_lossw<


total_lossl�?


accuracy_1   ?h�y�]       a[��	Z�.-_��A�.*O

prediction_lossq=
?

reg_lossw<


total_loss�?


accuracy_1��>m�?(]       a[��	�.-_��A�.*O

prediction_loss   ?

reg_loss�w<


total_lossl�?


accuracy_1   ?�E/]       a[��	|)/-_��A�.*O

prediction_loss���>

reg_loss�w<


total_lossg{�>


accuracy_1�?^1u]       a[��	 _/-_��A�.*O

prediction_loss�z?

reg_loss�w<


total_lossMW?


accuracy_1=
�>$�MO]       a[��	b�/-_��A�.*O

prediction_loss�?

reg_loss�w<


total_loss#�?


accuracy_1���>0̄]       a[��	�/-_��A�.*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>c�m�]       a[��	��/-_��A�.*O

prediction_loss)\?

reg_loss�w<


total_loss�8?


accuracy_1�G�>�|��]       a[��	��/-_��A�.*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>p�j/]       a[��	�
0-_��A�.*O

prediction_loss�?

reg_loss�w<


total_loss#�?


accuracy_1���>��&A]       a[��	�%0-_��A�.*O

prediction_loss�z?

reg_loss�w<


total_lossLW?


accuracy_1=
�>\���]       a[��	 B0-_��A�.*O

prediction_loss)\?

reg_loss�w<


total_loss�8?


accuracy_1�G�>6Nu]       a[��	�^0-_��A�.*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1��?�B�]       a[��	�0-_��A�.*O

prediction_loss�?

reg_loss�w<


total_loss"�?


accuracy_1���>�r��]       a[��	��0-_��A�.*O

prediction_loss)\?

reg_loss�w<


total_loss�8?


accuracy_1�G�>�T��]       a[��	��0-_��A�.*O

prediction_loss���>

reg_lossww<


total_lossc{�>


accuracy_1�?Ϥ|C]       a[��	��0-_��A�.*O

prediction_loss   ?

reg_lossmw<


total_lossj�?


accuracy_1   ?�$�]       a[��	�1-_��A�.*O

prediction_loss)\�>

reg_lossdw<


total_loss��>


accuracy_1�Q8?�+��]       a[��	�21-_��A�.*O

prediction_loss�G�>

reg_lossYw<


total_loss� �>


accuracy_1)\?�1�1]       a[��	�Z1-_��A�.*O

prediction_loss)\?

reg_lossPw<


total_loss�8?


accuracy_1�G�>��]       a[��	�1-_��A�.*O

prediction_loss   ?

reg_lossCw<


total_lossi�?


accuracy_1   ?�8�)]       a[��	�1-_��A�.*O

prediction_loss�G�>

reg_loss:w<


total_loss� �>


accuracy_1)\?�]       a[��	`�1-_��A�.*O

prediction_loss���>

reg_loss0w<


total_loss`{�>


accuracy_1�?�q;]       a[��	B�1-_��A�.*O

prediction_loss{�>

reg_loss(w<


total_lossL͵>


accuracy_1��(?��c�]       a[��	a2-_��A�.*O

prediction_loss���>

reg_lossw<


total_loss`{�>


accuracy_1�?�]       a[��	�42-_��A�.*O

prediction_loss��>

reg_lossw<


total_loss�=�>


accuracy_1q=
?��@p]       a[��	�Q2-_��A�.*O

prediction_loss�z?

reg_lossw<


total_lossIW?


accuracy_1=
�>C��]       a[��	�m2-_��A�.*O

prediction_loss   ?

reg_loss�w<


total_lossh�?


accuracy_1   ?l��/]       a[��	��2-_��A�.*O

prediction_loss�?

reg_loss�w<


total_loss �?


accuracy_1���>���]       a[��	��2-_��A�.*O

prediction_loss   ?

reg_loss�w<


total_lossh�?


accuracy_1   ?	?��]       a[��	A�2-_��A�.*O

prediction_loss���>

reg_loss�w<


total_loss^{�>


accuracy_1�?�Zҫ]       a[��	��2-_��A�.*O

prediction_loss=
�>

reg_loss�w<


total_loss��>


accuracy_1�z?"��%]       a[��	3-_��A�.*O

prediction_loss�z?

reg_loss�w<


total_lossHW?


accuracy_1=
�>�S�]       a[��	0G3-_��A�.*O

prediction_loss   ?

reg_loss�w<


total_lossg�?


accuracy_1   ?�z�]       a[��	i3-_��A�.*O

prediction_loss��?

reg_loss�w<


total_lossv?


accuracy_1���>���]       a[��	��3-_��A�.*O

prediction_loss   ?

reg_loss�w<


total_lossg�?


accuracy_1   ?�l
M]       a[��	�3-_��A�.*O

prediction_loss���>

reg_loss�w<


total_loss���>


accuracy_1��?$��?]       a[��	`4-_��A�.*O

prediction_loss�?

reg_loss�w<


total_loss�?


accuracy_1���>r֦J]       a[��	�!4-_��A�.*O

prediction_loss=
�>

reg_loss�w<


total_loss	��>


accuracy_1�z?��\)]       a[��	G4-_��A�.*O

prediction_loss�?

reg_loss�w<


total_loss�?


accuracy_1���>ɚ[]       a[��	mo4-_��A�.*O

prediction_loss�G�>

reg_lossxw<


total_lossz �>


accuracy_1)\?��7 ]       a[��	q�4-_��A�.*O

prediction_lossq=
?

reg_lossnw<


total_loss�?


accuracy_1��>���]       a[��	��4-_��A�.*O

prediction_loss��>

reg_lossdw<


total_loss�=�>


accuracy_1q=
?qD��]       a[��	/�4-_��A�.*O

prediction_loss�?

reg_lossZw<


total_loss�?


accuracy_1���>�*�L]       a[��	��4-_��A�.*O

prediction_loss   ?

reg_lossNw<


total_losse�?


accuracy_1   ?�{]       a[��	��4-_��A�.*O

prediction_loss��>

reg_lossDw<


total_loss�=�>


accuracy_1q=
?����]       a[��	�5-_��A�.*O

prediction_loss�G�>

reg_loss;w<


total_lossx �>


accuracy_1)\?�fl�]       a[��	�25-_��A�.*O

prediction_loss��>

reg_loss0w<


total_loss�=�>


accuracy_1q=
?�p&�]       a[��	UK5-_��A�.*O

prediction_loss)\?

reg_loss(w<


total_loss�8?


accuracy_1�G�>rp:]       a[��	Dk5-_��A�.*O

prediction_loss��>

reg_lossw<


total_loss�=�>


accuracy_1q=
?�>�g]       a[��	<�5-_��A�.*O

prediction_loss���>

reg_lossw<


total_loss���>


accuracy_1��?�<i�]       a[��	��5-_��A�.*O

prediction_loss   ?

reg_lossw<


total_lossd�?


accuracy_1   ?���]       a[��	��5-_��A�.*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>T��]       a[��	+�5-_��A�.*O

prediction_loss\��>

reg_loss�w<


total_loss$H�>


accuracy_1R�?Y��]       a[��	��5-_��A�.*O

prediction_loss�z?

reg_loss�w<


total_lossEW?


accuracy_1=
�>�*]       a[��	'6-_��A�.*O

prediction_loss   ?

reg_loss�w<


total_lossd�?


accuracy_1   ?S�h�]       a[��	�*6-_��A�.*O

prediction_loss�Q�>

reg_loss�w<


total_loss�
�>


accuracy_1
�#?�4d�]       a[��	cD6-_��A�.*O

prediction_loss�?

reg_loss�w<


total_loss�?


accuracy_1���>>@�}]       a[��	d]6-_��A�.*O

prediction_lossR�?

reg_loss�w<


total_loss��"?


accuracy_1\��>�mZ]       a[��	�t6-_��A�.*O

prediction_loss   ?

reg_loss�w<


total_lossc�?


accuracy_1   ??<<q]       a[��	��6-_��A�.*O

prediction_loss��>

reg_loss�w<


total_loss�=�>


accuracy_1q=
?����]       a[��	�6-_��A�.*O

prediction_loss���>

reg_loss�w<


total_lossT{�>


accuracy_1�?&v�]       a[��	��6-_��A�.*O

prediction_loss��>

reg_loss�w<


total_loss�=�>


accuracy_1q=
?�`�]       a[��	h�6-_��A�.*O

prediction_loss   ?

reg_loss�w<


total_lossb�?


accuracy_1   ?n)��]       a[��	~ 7-_��A�.*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>J6r.]       a[��	�!7-_��A�.*O

prediction_loss\��>

reg_lossvw<


total_loss H�>


accuracy_1R�?��~�]       a[��	l@7-_��A�.*O

prediction_lossq=
?

reg_losskw<


total_loss�?


accuracy_1��>����]       a[��	�X7-_��A�.*O

prediction_loss)\?

reg_lossbw<


total_loss�8?


accuracy_1�G�>Ҁ*�]       a[��	�{7-_��A�.*O

prediction_loss
�#?

reg_lossXw<


total_lossk�'?


accuracy_1�Q�>V�}1]       a[��	Օ7-_��A�.*O

prediction_loss�z?

reg_lossNw<


total_lossBW?


accuracy_1=
�>�+�]       a[��	l�7-_��A�.*O

prediction_loss)\?

reg_lossBw<


total_loss�8?


accuracy_1�G�>���]       a[��	>�7-_��A�.*O

prediction_lossq=
?

reg_loss6w<


total_loss�?


accuracy_1��>f�nv]       a[��	��7-_��A�.*O

prediction_loss��>

reg_loss-w<


total_loss�=�>


accuracy_1q=
?<��;]       a[��	�8-_��A�.*O

prediction_lossq=
?

reg_loss#w<


total_loss�?


accuracy_1��>�8��]       a[��	�8-_��A�.*O

prediction_loss�G�>

reg_lossw<


total_losso �>


accuracy_1)\?�1t]       a[��	C98-_��A�.*O

prediction_loss)\?

reg_lossw<


total_loss�8?


accuracy_1�G�>�[r]       a[��	@Q8-_��A�.*O

prediction_loss�Q�>

reg_lossw<


total_loss�
�>


accuracy_1
�#?�="�]       a[��	�h8-_��A�.*O

prediction_loss���>

reg_loss�w<


total_lossO{�>


accuracy_1�?Y��]       a[��	�8-_��A�.*O

prediction_lossq=
?

reg_loss�w<


total_loss�?


accuracy_1��>��wc]       a[��	 �8-_��A�.*O

prediction_loss=
�>

reg_loss�w<


total_loss���>


accuracy_1�z?��Ho]       a[��	�8-_��A�.*O

prediction_loss   ?

reg_loss�w<


total_loss_�?


accuracy_1   ?���X]       a[��	��8-_��A�.*O

prediction_loss   ?

reg_loss�w<


total_loss_�?


accuracy_1   ?-�?�]       a[��	-�8-_��A�.*O

prediction_loss���>

reg_loss�w<


total_lossM{�>


accuracy_1�?�߼N]       a[��	~9-_��A�.*O

prediction_loss�?

reg_loss�w<


total_loss�?


accuracy_1���>�Uz