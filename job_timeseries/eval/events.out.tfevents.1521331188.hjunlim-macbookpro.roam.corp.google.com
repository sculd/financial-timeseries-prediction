       �K"	   �j��Abrain.Event:2֍z��     � ��	 t5�j��A"��
�
-global_step/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@global_step*
dtype0*
_output_shapes
: 
�
#global_step/Initializer/zeros/ConstConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
global_step/Initializer/zerosFill-global_step/Initializer/zeros/shape_as_tensor#global_step/Initializer/zeros/Const*
T0	*

index_type0*
_class
loc:@global_step*
_output_shapes
: 
�
global_step
VariableV2*
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container *
shape: 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
g
truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"
   �   
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ↁ=
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
T0*
dtype0*
_output_shapes
:	
�*
seed2 
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
_output_shapes
:	
�*
T0
n
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes
:	
�
~
Variable
VariableV2*
shared_name *
dtype0*
_output_shapes
:	
�*
	container *
shape:	
�
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
_output_shapes
:	
�*
use_locking(*
T0*
_class
loc:@Variable
j
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
:	
�
`
zeros/shape_as_tensorConst*
valueB:�*
dtype0*
_output_shapes
:
P
zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
i
zerosFillzeros/shape_as_tensorzeros/Const*
T0*

index_type0*
_output_shapes	
:�
x

Variable_1
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
�
Variable_1/AssignAssign
Variable_1zeros*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes	
:�
l
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes	
:�
i
truncated_normal_1/shapeConst*
valueB"�   d   *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_1/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
_output_shapes
:	�d*
seed2 *

seed *
T0
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*
_class
loc:@Variable_2
p
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*
_output_shapes
:	�d
a
zeros_1/shape_as_tensorConst*
valueB:d*
dtype0*
_output_shapes
:
R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n
zeros_1Fillzeros_1/shape_as_tensorzeros_1/Const*
T0*

index_type0*
_output_shapes
:d
v

Variable_3
VariableV2*
shared_name *
dtype0*
_output_shapes
:d*
	container *
shape:d
�
Variable_3/AssignAssign
Variable_3zeros_1*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:d*
use_locking(
k
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:d
i
truncated_normal_2/shapeConst*
valueB"d   2   *
dtype0*
_output_shapes
:
\
truncated_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *��>
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
T0*
dtype0*
_output_shapes

:d2*
seed2 *

seed 
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
_output_shapes

:d2*
T0
s
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*
_output_shapes

:d2
~

Variable_4
VariableV2*
dtype0*
_output_shapes

:d2*
	container *
shape
:d2*
shared_name 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:d2
o
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*
_output_shapes

:d2
a
zeros_2/shape_as_tensorConst*
valueB:2*
dtype0*
_output_shapes
:
R
zeros_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
n
zeros_2Fillzeros_2/shape_as_tensorzeros_2/Const*
T0*

index_type0*
_output_shapes
:2
v

Variable_5
VariableV2*
shape:2*
shared_name *
dtype0*
_output_shapes
:2*
	container 
�
Variable_5/AssignAssign
Variable_5zeros_2*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:2*
use_locking(
k
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*
_output_shapes
:2
i
truncated_normal_3/shapeConst*
valueB"2      *
dtype0*
_output_shapes
:
\
truncated_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_3/stddevConst*
valueB
 *�5?*
dtype0*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
T0*
dtype0*
_output_shapes

:2*
seed2 *

seed 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
_output_shapes

:2*
T0
s
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes

:2
~

Variable_6
VariableV2*
dtype0*
_output_shapes

:2*
	container *
shape
:2*
shared_name 
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:2*
use_locking(
o
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*
_output_shapes

:2
a
zeros_3/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
R
zeros_3/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n
zeros_3Fillzeros_3/shape_as_tensorzeros_3/Const*
T0*

index_type0*
_output_shapes
:
v

Variable_7
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
Variable_7/AssignAssign
Variable_7zeros_3*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:*
use_locking(
k
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes
:
��
MatMul/aConst*�
valueݴBٴ	�

"ȴ�7(D\��@R��@��MA���A�(�A��A�QBff�A�Q�A�H)Dף�@��A�A=
�A\�B�(BףBffB�G	B��(D��̿H�*@   A�(�@�zxA�(B��Bq=B  B�`'D�����(��=
W��(�?��Y?  A{�A�G�Aq=�A�K)D�p�@���?
�#=��@q=A�QA)\�AR�B�QB33)D\�¾�G�@�G�?{���{@�(Aq=A�Q�A33BR�)Dq=�?���?��A��E@R��?�Q�@�p1A�A���AN'D\��\���R����z��q=���p����i�=
�?)\? P)D�z Aף��ff�>)\�=��@H��?�G�=�(�@)\A� ,D�(,A�Q�A{A)\3A�G-A  �A�GA��-Aq=rA�k*D\����@)\GA33C@�(�@  �@R�BA�z�@�G�@�E+Dq=Z@H�:�H��@��}AR��@ףA��@�GyA��A�B*D����
�#�=
��\�r@�=Aq=@��@=
w@�z8A.*D
ף�
׋���u��G��{^@  8A��@H�z@\�b@��.D�p�AH�A��dA�Aq=6A33�A�p�A�(�A�̴A��0D  �@�p�AH��A�z�A���A��A33�AR�B�(�AH1/D�(��)\�?ff�A
םAH�zAR��A�(LA�(�Aff�A)-D�G	�)\_�R����7Aff2A33�@�((A�@=
oAR�,D���q=��Qt��Q�\�"A�pA�G�@33A�7@��0D\�rA��]Aף�@�G��z�@\��A  �A���AH��A�k1D��h@ff�A��A\�A��a@�z$A��A��AR��A�~1D���>�(|@�̘A�Q�A)\A�u@�G)A{�A��A�3D��A��	A
�CA33�AR��A{�A{BA=
�Aq=B �5D�A�(�A\��A��A�zBq=B
��A�̪A���Av6D�?�bA��A(~�A���AG�B״B���A���A\7D��@��@)\{A{�A�z�A���A�p,B33'BH�BRX;Dq=vAq�A=
�A���A��B��B)\-B  jB��dB=J;D�Ga�R�rA�X�A�G�A=
�AR�B��B�z,B�iBq]<D���@\��@�AH��A��AR�B��-B�/B�=B  =D\�B@H��@
��@{�A4�A  �AH�B{:B�G;B �<D   ���?H�@
ף@{�A4�A  �AH�B{2B�q>D���@���@�A��IAffFA�Q�AR�B�	B  .B\?=D�G��)\@H��>��a@\��@��@  �A���A���A �>D�Q�@�Q8?   A  �@ףA�pUA��QA{�A��	B3@D���@��0Aף�@��XA��8A�piA��A)\�Aq=B��?D��̾�̤@\�*A
׻@ffRAff2A=
cA��A�(�A�u=D���)\#������Y?�(|���U@��?�(�@H�
A�:D  \��z������G��ffN����\�&�\�F�����#:Dף�>�zT�R���������H�F���=
�=
?���;D�p�@�z�@���)\��\����Q0��Q����$��GA� 0:D)\��\�B>��(?�pQ�33��ff��  ��
�C�q=���>9D�Gq�  (��e�=
G�H��)\��\����(��{���,:D{n@��L�����)\>�(?q=R������̺�ff��\�<Dף,A�(hA
�+A�Q�@H�.Aff6Aff�\�B���H� �-D��q���F�7�  G�a�q=F�)\D�)\{��L�r*D)\S�ff��ף{�l�
�{�Ha��{{�33y���)D�G�ף��ף��\��q=y��(�¸��®ǃ�
ׂ�f*D=
�?�Q��ffj��G��3����r��̀B�k��WS)D(C���������'`�¬ˆ��~�F��	[�¤p)D�d�>��%���h���������u��H��H�|�����(D=
'���	�ff���Ga��z���̞���������*D$��@
h"@��?@b�V�2U�?���=k��b��΀���+DW[�@=
GA�GAԚ$A��@R�A���@����̇���+D��>Q��@�ILA+�"AQ�)A-�@��A��@����q�*D���(���d*@���@ff�@��@=
'@�G�@�?n*DR���V��
׻�4�?q=�@�p}@�^�@)\�?33C@��&D��i��p}�Nb���£�s�R��z �q=*�K�"��S��#D�Gm��p��)\����33��H�����������%D
�+AH������p������z��$����A��k��y&D�7@��YA�(���}��z���$������\f�  �h�$D����~�"l�@���U�� A�����^K������3##D7����U��'���>�i�)\���G���x��(¤�!D�G���<�����Q|�����H��
�	����ˡ"�!D
ף�
׫���=�����}��������(
¸�q'Dq=�A���A\�zAA�A
�@�¥@R�~A��?�(X�~'D�G�?�Q�A��A)\�A6�.Aq=�@{�@�p�Aff6@�w&D33�������AH�A�UA:#�@������5@�GYA�&Dff@   ���u��Q�A��AR�vA6�A�z@{�@�I%Dq=��=
���������_Aq=^A��	A�1@  ��q�'D��A)\/@H�@�p=?   @q=�A���A�G�AA�:A��*D)\OA�(�A33{Aff�A33[A)\wA��BףB���A��,D�A�p�A���A)\�A�(�A)\�A�p�A
�2B�2B��.D)\�@33kA�G�AH�B33�A  B33�A�G�A��LB �-D�G�=
W@�G9A�Q�AffBq=�A=
�Aq=�A�Q�A��,D=
��H����(\���@\��A=
�A�z�A�G�A�z�A{�+D�pM�������(�q=���̄@H�A)\�A�̞A���AR�,D
�S@���=
׃����\�B�R��@)\�A
��A�G�Aq+D�p��=
g�q=��ף0�\�b�����q=
?  XA�z�A�Z+DH�?R����������G�33O�=
��  �?)\kAR8.D)\7AR�JA  �@��A33�@ף�?R�����@)\SA�c.D{.?q=BA��UA���@
�A���@
�@)\O��p�@ͬ/D�z�@q=�@q=�A��A�9A{nAR�:Aff�@\��@�}/D�<��@���@yX�A��A�T-A�JbAc�.A���@�14DΈ�Aף�A�¹A33�A�pB�GB33�A
�B  �A{D4D�z�>�ژA���A{�A��A��B�pB��A  B
4D��u��+��,�A�G�Aff�A
׹A��
B��B
��A
�4D  @��?q=�?�,�A�G�Aff�A
��A��B��BR�2D)\��)\��{���̤�6�ZAH�NA\��A  �A��A `1D�(����M���)��9��z4�w-�@���@=
?A��IA�71D�G!��Q��
�W�
�3�33C�\�>���@�p�@��4A��0D�%��pM��p�\����]��zl�
�g�%u�@��e@� /D�������
����y��̲��̠��z���(��u��R�0D
��@��?R�����  ��o��K�=
[�ffV� �4D��}A��A��A{fA  \A
��@
�c?��H@�@��4D=
�>�Q�A�G�A=
�A��lAR�bAף A��?
�c@~8D��dA�kAR��A
�B�p�A���A���AR��A��yA �7D{N��G1A  8A���A��
B��A=
�A  �A���A3S7D�����p��{A�� A)\�A�(B{�A�p�Aff�A��8D-�@P�W@P�>+gAT�mAm��A�nB���A���A�^9D+W@H�A)\�@ף`@�z�A
בAffBH�%B��B\�8DR������?{�@��@�G@\��A��A�pB��B �4D���
ב�T�m��� �  8��k�=
׾    ��}A�;5D=
�?��p�ff��S������ףP��G�?=
�?)�2D
���� �ff���Q���l��H���z��q=����{�1D�w���Y�H�>�)\���G��Nb��
ׯ��p��33��� 0D����H�*�)\��������H�n��Q��������/D��ѿ33��E��z��=
���z¤p���¤p�� `9D�B��B�p�A�z�A\��A  �A�G�?
ף<�rX@u<D�GEA
�KB�GEB=
*B\�B33�Aף�A�p]A��EA�j<D
�#�R�BA33KBףDBff)B��B���A)\�AH�ZA��;D�(<�ffF��A�p?BH�8BףB�(Bff�A
��A�k<D��<@
�#<���H�BAq=KB�DB�p)B��B  �Au<D��>ffF@
�#>    �GEA
�KB�GEB=
*B\�B)�<D{�?ף @R��@�G@{�?=
cA�GSBR�LB�z1Bf�AD�G�A�(�A)\�A���A�p�A�(�AffB�u�B.�BH�=Dף���U@{�@H�@ףA33�@{�@�(�A��`B\o?D=
�@����� A\�>A��@A�(pA�AA\�>A���A�@D�@ff2A�­��gAR��A��A��A  �AR��A�|AD��q@)\AH�nA33ӿ{�A���A�(�A�¹Aq=�A�WDDR�6A33sA=
�A���A�QA�p�A�Q�A��A\�
B�CD��<��A  DA�p�A33�Aq=�@
��AR��A���A�,CD�(ܿ�p��  �@�z(A)\oA�p�A33�@{�A���A
�CDq=
@�G�>ף �\�A=
KA���AR��A�Q�@)\�A{�DDR��@
��@�̬@��@��]A33�Aף�Aff�A�CA�8FDq=�@�z A=
CA�'Aף�@��A�µA33�A�zBf�ID)\_Aq=�A��A33�A�p�A
׫A��BR�B�p$BqFDq=f��(ܾ�z�@��A�(<Aף AH��@{�A�Q�A��FD�zD@�5���(@R��@R�JA�GmA��QA\�"Aף�Aq�GD�@  �@�z��q=�@q=>A�̆A{�A�Q�A�peAH�ED=
��Q���Qؿף������<@��@�!A��AR�ED
�#��G��̬�����ף0�
�@ף�@H�A��GDffA�(A)\�=  �@q=�@q=���z�@)\?A)\�A�LHD�µ?�1AH�&AR��?�p�@
�A�̴���A{VA)�ID��@��@�z�AR�vA)\�@\�*A�[A��(���TAB`MD%uA�n�A�ʭA���A;��AZ�A���AZ�A�vjA��QD�̈A�B��B�KB`�ABV?Bu�B�K,Bu�8BR�OD+��A��A�p�A���A  %B�p"B)\�AffB%&PD㥛?�A���x1A}?�A+�A�CB/�)B�M'BD�BH�OD^���)\>!����A Aף�A\��A���A\�%B  #B{�PD33S@V@�(\@-r�VUA=
�A�zB�(B��2B
�NDR���q=z�P����Gq�m�/�Zd�@)\�A�G�Aף�A�SND33��(�R����&��q=��9�T���s@���AH��A�bMD�Gq�q=���zT���`�0��p�����t=��uAT�KD����� 0���T��$��w���Z������j���v����IDNb��  \��(��\���q=��
׽��r��R������{�JD��a@�>��#�
�_��Q��  ������?5���z��P�KD��X@B`�@;��>����ˡ)��nN�`���~�������JD�\���u�{^@!�B��z$���`��̂��z��{��דJD33s�j����G���G!@�|��3�  p�ff��{���hJD�+�)\Ͽ-��=
׿���?/��ff>�R�z����KD��1@=
@�z�?�n��̌?�(�@`����q=N�=�HD�(�)\��������	���@�\�
��(���E:�=
���FDP��#ۑ�d;w�����u����(��V��ˡY�y��)LHDb�@�(���3�33�����!��IX�{"�33���BKD��=A�ЖA{&AR�?��Y@R�.@
��?�տ�(�?�ID����Nb�@X9@A!�b@����1\��t��#ۡ���{�FD�2���
�����a?H����̊��i�
�s����3�CD�Q@�L7���������?52�H������R���{��f�AD33��¡��������R���9����z���G��(iBD333@�̬�)\��j��\���Q���M��{��{�ED�zHA�GuA�(�@�z��-��H���(0��A���G���ED=
W>
�KAףxAH��@����|�33����,�os�3�ED{����ffFA33sA  �@ף��L7�����q=2��{CD���)\�  ����@\��@)\��q=R�-��H����WFD=
7A�zT@R�>@�(L@�{A�(�A�%A��ٿ�OM�ףCD��,��G!?���\���
�����@R��@H����(H�z?D+������ ���������� ����;���������GD��B�z�A  �@��A�A�A=
A�±A�(�ALHD#�y>��B�n�A���@�x�A%!A��A��AF��A��ID�&�@�z A��'BR��A�zhA���A�̎A{�A�A]KD^��@�pUAXYA�(>B�&�A��A'1�Ad;�A��AfFJD�S�����?��A�Aw�,B�Q�A�{A)\�Aff�A�ZHD����D�@�)\��+�?ff�?%BH�A�� A��A��LD��A)\AZd�@\�2Ah��A��A��TB  B��A{$ND�Q�@33�A�wA#�1A)\�A��A���A��jB=
(B�MD�z��)\�?{�A�G5Ad;�@�zHA+��A�z�AbZB
gODH�A�G�@��(A��A{�A}?�A��A���A���AnSDH�A�Q�A33�A�G�A331B�zBbB�GB�l6B3cWD�G}A��A�z$B��B��)B�pB��QBNb@B��VB�NVDq=���(8A���A33BףB�Bq=_B�@B�/BRhZD33�A�GAA�G�A{0B��TBq=DB�GZB�k�B\�B��YD��	�  dAH�A{�A�z'B33LBף;B�QB��B�\D  A33�@  �A�p�A=

B�zJB33oBף^B�tB�a[D��,����@��y@ff�A�A�z�A�?BffdB
�SBf�]D�A
��@��yA�WA���Aff�A�%B��eB
W�B��dD�Q�A�pBףBף1B=
)BףjB)\YB
W�B\��B��aD�Q@��(�AR��A��A\�B���A\�:B�G)B��hB�KcD��@R���ף�A33�A���A��B33B��OB�>B�!cDff&���@����p�A  �Aff�A33B��B33MB�bD�ǿ�p�ffF@R�����A��A���A��B)\B�Z^Dff��H��{��33g������z@q=>A=
A��A��\D�(���p���������ף��ff���Q�@
�#@{TZD���̀����
�¤p�ff���G*��z\�R���HQ\Dff�@\�¿ff�����{���G���̴��z
�\����YD)\O��Q���g�H��ף�H� ��z#�q=��Q>¤�YD��@�((�
�#��z@��G��
��{��¤p���YD�~�>�:@�� �+���&9������}?���R�XDˡe��QH�{.�q=Z�{��\�r��Q��)\�#�f�VD�z�T�A�\�:�)\�)\���c����\����z?���VD��?�����I8���0���	�\�����Y�R��������(XD{�@�G�@)\?��~��
���H�j�=
��H�
�33����[D�pqAq=�A=
�A��AA'1A�AR�6A�ſ��@�^D
�'Aף�A�(�A���AR��A�A��A�G�A33A��[D\�&�
ף=R�rAH�A��AH�BA�x	A��A  8A��YD\������  ��H��@�zDA{NA33�@oC?�(�? @[Dף�@
�3��S�R�.���EAff�A33�A��A%�@��ZDH���
�C@ף��H�r�{��ff&A�puA=
A��@ �]D)\3A  A�QdA{�@{~�ף�@H�Aff�A33�A�'aD��eAף�A���A��A�z�Aff&A��A��B�#B�+aD��u=H�fA��A�p�A���A���A)\'A���A�(B\/]D=
�{~��G��33A��@�(LA�@)\���Q�@�;\D
�s�  �����q=���z�@�{@33A)\�?ף��rYDq=2�33o����ף�����  ��R����(��Q �͌XD��e��k��Q����	��	�ff���p���,������iZDff�@=
w@�����p1�q=��������I��뱿ffV�{D[DH�Z@��-A���@=
w��p��H��ff��H����@�\D33S@=
�@R�bA�G)A)\�
׋��z��  ���(��R�]D�Q�@��,A�cA�p�AR��Aff�@��H@��L�
�K�ד\Dq=���Q�?��@\�
AH�A�QHAף�?��������]D
׃@�������@��A�zLA
סA��A  �@�Q�?\__D���@H�2A��@��QA)\�AR��A�Q�A���A��HA�EbD��9A�G�Aq=�A��A���A�(�A��A\�B33B\ObD��>  <A�z�A�p�AH�A���A)\�AR��A�(B\bD  @?fff?  HA�z�A�p�AH�A���A)\�A)\B\fD  dA  pAffrA  �Aq=BR�B�pB�zB�,B �fDף0@{�A{�A�G�A{�A�GB��"B�zB�*B),gD�Q�?ff�@���A���A�̜A���A=
B�)Bq=B��eD�����Y���!�T�YAT�eA�IhA���AF�B'1B3�eD�l���z��ff���Qؿ��HA��TA)\WA�z�A�z B��cD
������(`��E����{�@{�@H�@��AF&`D�Oa�����{��j���L7���"���E��E
�;��ͬcDˡaA
ף<33��+��
�_���D�ף�R��@R��@3�aD�����v�@�(��  l���|������������(���aDfff��������@����ffz������Q���̠�R�����]D33s��̀�  ��5^�
׷������C�����33�fV]Dff�ff������������3�ף�����1	�)\�H�\D�E��­�=
��q=���p��}?e��G����Z��^Dff
Aq=�@{N@�?�{N�q=��-���{��=
�� �]DH�J�)\�@��@��L=ffr�ff������h���p��bD�A��TA���A���A�(�A���?=
�>)\��m��@{�bDff@\��AffzAff�A�©A���AH�j@�G1@�(�� @bD��ȿ�G?  �A�GaA
׵A33�Aff�Aff@���?{$`DH��   ������A�̴@��dA�3A��A\�����^D�p����e�R�~��Y���9@q=��{A���@��<@=*_Dף @q=z��pE�\�^���8���@R��?q=&A���@=�^D  ���u�����pi��G����\�q=*@\��q=A�`]D�̜������z����0�����z�����)\���� P\D�Q��\��\�6�ff��u�  ��\����·�  ���sYD=
7�33{��̤��̶�R���{�����=
�ף	�=�VDff"�R�������  ��  ���ף�)\5�ף;�:VD� 0��nN�j��������������5^@iXD;�A��@�����y�H��{��{��  ��)\���ZXD�k�'1A�Q�@�z���G}�R�����������
���UYDq=z@�k@��FAR�A����R�>��p��ף��ף��qmZD�(�@ףA�� Aj�A��`A��y@�G����<�����q�YD  `��Ga?�G�@��@��TA��(A���>ף0���t���XD�zt�q=���(<��Qx?�p=?F�A)\�@H�Z���m���TD����z���z���p���Qt�  x��A���(��G��3#TD��̿R����G���G��q=�������̈�^����1���UD�(�@���@)\/��zl�q=��ff^�
���#�/����SD�z�33ӿ  P�����z���z���p���(��  ��RWD�WAff�@�G=A�#A�Q���G��GU�33��G����YDb4A;��Aˡ�A��A;߫A�Χ@�E�?/���t@��[Dˡ�@�p�AףB=
�A{�A�G�AR�RA��A33�@�[D
ף<�E�@���AR�B33�Aq=�A�p�A=
SA��Au^D��Am�A)\�AZd�A��$B�~BB��B�O�A�*^D-�?�pA��A�I�A�Q�A{'B��B�z B{B)`Dף�@-AH�A=
�A�r�Aq=B�(EB=
#B\�>BD_DbH����@�Q�@w�WAbXA�p�Aj�B�8B7�BZD^Dw��m������>�ʁ?��A� A�x�A��A�(B��]D�l��^���H����h�+���H�A33A�A=
�A
�]D33ӿ�O=�+����G!�
�#��������@���@��gA{�aD��A��xA1dA�$A�(�@ffjA�AtA��A{�A��`D���zDA{*A�&A�n�@��@�AB`%A�z�A
�aD=
�@=
W�  �A��uA�`Aj� A�p�@=
gA`�pA �]DH��q=>�\�����>�G��Z$�J��=
�H�
�õ]D{�>�(����8�
׃�H�:?�k����+������N[D����Q�=
���G��R���{��z(�sh=�X}��[Dףp?R�
��G�����¡�33��{���p�5^.��[D
��=
W�33'���!�����  ���p������5�=�ZD����ף�����H�.��p)�����
׳��G��33#�B�]D�1A��)AXANbA��'>%?ـ�-6�+���\�]Dj|?�GAA��9A�A�(,A33�?R��?��q�ff&��[Dff>���.��Q8>�������둿  ,�\�&��(��fZD)\o�q=z��vj�
�c��G��q=���(��
�g�ffb�\oXD�����%�  �����R�"�ff*�H�F�
�7��̨��jUD�A��p��)\���G	�V���µ�  ���z��
�UD�(@{����
ס�� ��&��ff��q=���z��\YD{FA�iA   @���33��  ������p��ff�\�YD  p@=
�A\��A  �@�Q��33��  ��X9p�H�z�
[D
׃@
��@  �A��A��%Aףp@
ף<{>��M.���\D�̼@�Q A�Q\A33�AR��A�(�A\�A�p�@)\��ہ`D�&�AZ�A�O�A�O�A�(B�o1B�'B�n�A��A�2^D�����@��JA�n�A�n�Aj�B�~B�n�AP��A=�|Dd;�B��BffC�LC�kC�+C͌Cq�C��C3�zD����B���B{��BHa�B\OC\C�pCH�C �{D��l@���yi�B'��B���B��C�C��
C�#C�D�G�AH��A  �A��C�aCffC�LC�kC�+#C�1�D���ff�A  �Aq=bA��C�
�B=�C�pC\�C��}D��%����33Aff>A\�r@���B�W�B�0C
C �}D��L���(��G��   A33;A��e@yi�B'��Bq�C���D��A�Q�A=
+A\��@���A��Bף�A-rC6
Cq̀Dq=��R�vA�sA
כ@�p}�)\�A���A{�AB�C�ҀD
�#>����GyA{vA���@�zT�ף�Aq=�A)\�AHa�D\�b��QX�33�ף@A�p=AR��?
׋��Q�A��A�C�D{n�=
�����{"���1A\�.A)\?����H�AHaDH��ף�����ff��k�ף�@q=�@����q=� �D�p�?=
W��G��\����p��
�W�  �@���@333���{D��l��Y��G��R���=
���³��Q������  ���~D�GA=
������G!��(0���h�q=f����
׃?å|D{����(@\�B�H�.��Qx��������ף��33����|D�Q�?  ���u@�/�
���Ge��(t�ff�����
�D��A=
�A��A�(�A�A33/A��@�­@���?\g�Dף0@���A��A���Aq=�A�GA)\[A��A=
A�1�D��տ��?q=�A�·Aq=�AH��A��,Aף@Aff�@��Dף ��k��k��(�A��A�QxA�̼A��A�z Af&�D33������z �ף��R�VA��iAR�A  �A��@דD��8�
��  4�R�N�\�"��z(A�;A���@��eA=B�D�(<A��A�G!@\�?�z�����?�Q�A
׻A�Q�A{��D���GA{�@{�>��տ��U��z�H�Aff�Aý�Dף`A��=A���A
ץA{fA��EA33+A)\WA��BM�D�GA�(�AR��AffB
�BH��A���A�p�A��A��Dff�@�p�A���A�Q�A33#BףB�z�Aff�A=
�A�r�D�Ga�=
�?\�ZA���A�(�A�B\�	B�Q�Aq=�A\σD���{
��{���AR��A�G�A� Bq=�A�p�AR��D�(��\�������̤�\��@���A��A���A�z�A�E�D�p�@��l@�z�����k�  DA�Q�AH��A�zB�*�D=
W�\��@=
7@  �ף������\�6A���A�(�A׻�D�zHA=
;AH�Aq=vA�z$A�Q�@)\7A��A��B{ĆD�QAff�A��A=
�A�G�Aff�A�zpA
םA
� B��Dף�@ףlA\��A
��A33�A�p�A\��Aff�A  �AH��D
�����Y��p�@���AH�Aq=�A�z�A���AH�bA���D\���33k�H���Q�=��IA�z<A���A�wA��%AH��DR��  �������̊��G���\@=
'@���@���@\�D=
CA�GA@���H�:����=
G@q=zA��lA�A���D�G��A��̽������l��z�
�#��QHAH�:A ��Dq=nA�Q<A��AףlAR��@�Q�=��@{nA�G�A
ǇD�(�?H�A
�SA�p�A{�AH�A��?�GA�̂Aד�DfffA��}A{�A��A�QB�G�Aף�A)\A
׳A3c�D�gA=
�A���A��4B�z(Bq=YB\�4Bq=B��A���DףP��3A���AR��A��'B�pB33LB�'B33
B���D
׳@=
@R��A���A
�Bff>B��1B�bB  >B�̋Dףp?���@33S@q=�AR� B��B�(BB�5B�pfB���D=
�   ��Q��z���pA��A��AffB��B��D  4A
�@  P@��A��@R��A��	B
�BffKB�#�DH���q=j@����H��
ף?�p��  HA33�A���A  �D���)\{�R���ffV�)\G�H�����!��(�@q=�A=��D�EAff�@��X���@q=��)\�)\�@{@���A�[�D�Q�A�pB���A33�A33�A��A33�A�(�A{�A\��D�@�G�A��BH��A�(�A�(�Aף�A�(�A��A�G�D
�/���	�R�NA���A���A�zAq=�A�p=A�zLA�E�D��u���0�H�
���MA�p�A�z�A�A�¥A�z<A\_�Dff���Q��  ��{~���@
׏A��A�G@�QXAͬ�D)\�A�A�G�AR�Aף(A�Q B��1B�B��Aד�D�G���A��A=
�A�z�@�(Aff�A�z.B  B
��D33�@q=�@��B�Q�A
��A
�oAH�A��BH�LB�ЏD{���z�?ף�?ff�A�̢A�Q�A��AR�:A
�Bfv�DR�RA)\�@�GqA��dAH�"B{B
�B�³AR��A���D���?
�gA��A33�A��yA�((B)\B�B�Q�A3�D�p��fff��A�Qx@�7A33+A�zB)\�AH��A�z�D{�$��z���\���Q��\�����Y?33��\��D�̴��zk��(�H�y�33E���[�\�=��@�  ��R`�Dq=�A��Q�)\�=
(�"��(��
��H������q]�DR���{�@�G��R�T�ffh¸c¤p.�33E���&�{�D���A{FAףB��8A���=
���z��q=r�ף��
��D�Q6B��B
�gB�z�B\�dB���A33A�(�A��Aí�D�(>�H���{�AR�&A�� B��A)\��R����(��E�D)\�@q=+®G1@���AffrAR�B�GeA���H���f�D�G�@�QA{��@�B��AH�'B���A33�����D{�Aff�Aq=�A{��\��A�(TB��B��jB�B.�D
׋A��B�B=
0B��a�33(B=
�BR�YB�k�BÝ�D�Q���OA
��A{B  Bף ��(B�B�GB�ŔD  �A
�KA���A��;B{PB  cB)\A�([B��BV�D)\_��(\A  A
��A��-B�BB=
UB=
�@33MB���D)\SA�A���A��A��B��bB��vB���B�p�A�Y�D��9@H�A��IA���AH��A)\+BffnB�G�Bq��B)\�D�GA�/A��A���A��B��B�KB)\�B�p�B�J�Dף����(@ffzA\�BA�G�A33�A�)B\�lB���D\��?ף����?R��@)\�AH�^A�p�A)\�A��0BHy�D
ף����?H����p}?��x@�̈A��YAH��A���AR��D�AffAR�"Aף�?)\A��IA\��Aף�A�QB ��D��@��Aq=~A�G�A��A���A�̠Aq=B\��Aq��D{.>��@=
�A�z�Aף�AףA���A�(�A��B�f�D��u�q=J�ff�@R�vA��qA���A�GA�G�A�z�A���D
�c�ף��33����H@��=Aף8A��TAף�@��MAV�D)\O��(��
׋��z�����̌��µ�33�>=
�ݗD�CA�p=������Q��H����@��1A��,A�IARH�DffV@�yA=
'@33s��z��R�޿  �@�gAffbA�s�D�̬?ff�@)\�A�p}@���>)\��Ǿ33�@�}AHٗDq=��{^�������AA�(\��p���(��R�����@=r�D{N�ף �{����U�{A\����z����	�H�����D�G9���l������(��R�n���,�\�z����p����D���@ף�������D�)\/������@���\�>�졔D���H�.�{��
���ff����������{Z�R������D{��  
����{���'�q=;�
�5¤p(¸��3�Dff�A\�B@33s��z�������{���G���z��Rh�D�G�A
�B���A�p}@�Q4AR�����a����  �� �D�(@����@���A=
#A�� ��p=��E�ףx�H����D�����a���@H��A��Aq="��5�\�f�=
��H�D��a�����������
�#Aף��{������q=��ý�DH�����®G%��QU���m�=
���zE�q=(�
אDq=J?\�����¸"��(R��®Ga�R����QB�
�D  ������GL�Ha��\���{����p��Q#�)\d��H�D�����1�.���¥¤������ף��{j���D
׻@�=��G¸�\�{���33��R8��f��RP�D�z���k>���
�0��-�\���L���z�� ���{܍D=
�A
�kAH�AH�:@ף���Q����@�{y�R���3�D=
W>R��A33oA\��A�QH@����ף����?�q=x� �Df����Ĥ�Ě)	�����ą�ď�� PD  ��fv��hĤPĚ���A���^ą+�R(D
ד������Ĥ��Rx�H�Ěi����ÅĤ`D=
�A�(DA
�_��e�RX� @����H1�q�ď�D�zA�G�A�Q�AR�����f����
G
�\��)�D�̢��-����@�Q�?�z��=���{��q]ĤpDH��q=��  ���7��������U�RH� 0��LD{�@R���R�����D�q=�@��L�ff���y�)l��,	D  `@=
/Aף @R�������A��\@ff����Ě�D33sA���A��A��A��H��z�@�(�A33�A33����D�z�����{�\��@ffV�����R�b��p-@�z����D{~�q=���G�\���{�?q=��)\������G�� `D�(�@q=
@33����L����>��@�Q���Q���(@��`D���{�\������=
3�{��  ��H��q=��)�D�zD���0���ff�����(d��(,��zT�  D�f�D�p5�\�f�33���(�����33#������̰�\�j���D����)\7��zh��(�����H���#����±�D��UA�SA���?=
��R��\�B��Q��\�����E�R�D����ff>A�z<A�G�>�Q(���)����)\��(���D��i@��@��xAH�vAH�@\��?=
��=
W�����wD����G���p�q=.A�Q,A�����h�{:�  ���DH�?��L�ff�>  ����EA�CA
�c?��R�"�{�D�(L���4�)\���D��(\���Ѿ����=�=
o� �Dq=��ף����}�q=��=
��ף��)\��33������
DH�
@���\�r�33[�H��)\k��G��
�3��;���Dff�@��	A���@)\�  ��\�>��(�)\���h@��D��<A{�A�p�A��}AffF@��@��̽33c@ff@�bD=
G@R�nA���A�Q�A�AR��@�p�@ף@@��@=�D�(��ף���G�@
�cA�G�A�p=A�k�q=
?
׃��GD�G��R�F����   @33A��1A���@R���  ��3D��Q����
�S�{"�=
�?{A��$A)\�@�����D�z(A)\A�p�@�p-����>)\CA�G�Aף�A{�A�D��?�pAA�Q4A)\�@33��H��?�Q\A�¡A��A�l	D��A��-A33�Aף�AףtA\�A�Q4Aף�Aq=�A��
D  �@��pA���A33�Aף�A�Q�A\�^A�(�Aף�A3�D��A��aA�G�A���A  BR�
B��A{�A���A��D�1A)\�A)\�A��	B�(B�G:B  7B
�!B�QB͜D�p�����@  pA  �A�z�A���A��(B�Q%B�(BףD�G�=����Q�@��qAH�A)\�A
��A=
)B��%B��D��?���?{N�33�@���A���A{�A�GBff-B�D����=
������z(�q=
?q=Aq=jA���A{�A��
D�G����ff��=
�H��ף�ףp���@��aA�p
D  @��G)����ff�����H��ף ��Qؿ��@�;DR�2AR�&A=
?=
���(��ף��=
�ף�?�A�D�̌?�QDA�Q8A�Q�?
׳������p���p�R�@��Dff�@���@�A�A�z�@�k�{.>)\�>�z��3D���G@�W@ףhAף\A�p}@ffV�ף���	� �D��L�R�N��(�?�z$@
�[A
�OAq=J@�̄�
�C� �D   �����3������­���@��@H��ffB�fD��Y?����ff��{&��µ�\���H��@H�@)\�3
D�� �ff��33s�  ���p���[�{J�Hế�p�\�	D�׿����(�{���z�����ףv�=
e��GI�=*DR� A��AR��>�z�?H����z���!�
׫�ף��\?D\��@  fA=
KA�z�@��@ף �
�S�������H�D
��?��@�z~A�cA�p�@ף�@H�z�
��R�����Dף�@��A{ZAff�A��A=
_AףlA�G�@��@�"D  �ף�@���@{>Aff�A��A=
CAףPA�G�@�{Dq=VAq=:A�G�A��A�(�A�BB�Bף�A�p�AH1D�z���CA�'A  �Aq=�AH��A��BH�B)\�A)lD�k?��u�ffRAff6A)\�A���Aq=�A�LB\�B�*D\�����̽�G��{BA{&A33�A�p�A{�AR8B\�Dq=�@���@=
�@��@���A���A���A  �Aף�A�D��5���@H�@�Q�@33�@��A��A{�A�Q�AED33@��?\�A�z�@��A���@�Q�A�Q�A�z�A��D�z�
ף�H�:�H��@q=�@��@\��@�A�A��D��E�H�j�{F��pQ�ף���G��
���������@
�DR��@�������p��{�������z���z$��D�zT����?�+�ףP�
�+�337��(���̴�)\����D��@\�R@  A��������(��H���p�?��u?��D��@�cAff.A��}A  `@=
�?R�^@�G1@�pA�CD������@�QPA33A\�jA33@
ף���@���?��D�(���h�{~@�G)A�Q�@�CA)\��p�
�#�{4D���
�C�=
W�\����G?\�"�H�@=
��(D�
GD�Q�A
�gA��@A��-A��A\��A  �A��A��eA)�D�p�����Aq=�@�(�@�@ף<A33�A�GqA�Q�A{�D�(�@�ſ  �A33OA�((A��A)\�Aq=�A��A)LD)\�?  �@
ף=���A�iA{BAH�.A�Q�A33�A��D��)\�q=�@ff���Aq=FA33A  AH�A�D���������Q������{�AR��@ף�@�zt@H�Dףp?)\��p�������z?H������AffAR��@�|Dq=���(�����
�3�������\�2�{RA�U@)�DR��?�e���(�q=��  �{�  @�R����iA3�Dq=~�fff���q=���������(������z��ffDff��R����̢�)\��
����G��R�������R���H�DR�
A{.@R�R�H�:�  ����|����)\��ff���HD{��뾸��ff���z��=
���������33 �\�D�̬?����
�c?����������q=��R����(����D�+�q=��)\'������=
���������(��)|D�(�@��\@���@\���H�@{ο  ��{��ף�� PDף0�{�@ף0@��@ף����i@33���������D�(��q=��R��>��\���ff"��p������\����#D=
GA���@
��@  LA�!AR�6A\�@)\/A�Q�@�)D�Q�=�zHA���@R��@�pMA\�"A�(8A�Q@��0Aq�Dףp���Y��p9AR��@ף�@ff>A�A�)A�Q�?�,D
�A�� Aq=Aף�A33wA�(lA��A��A�z�A
GD��@ffVA)\GA��HA���AH�A)\�Aff�A���A=�D��ٿ�p-@33;A�(,A��-A�Q�A�G�A�A�̼A�D)\?�{��)\��)\Aף�@��@ff�AR�rA�gA�"D���=�7�q=��
�#��GA�z�@)\�@)\�AףtA{�D�(�����H���G���Q���@�p�@�Q�@���A�D����p�������G��
��\�����@{^@
�c@ �D
ף�q=��ף�R��\�F���a�33���5�33ӿ{D�+�����  $��C���A��pq��Q��{F���X�
�D\�@
�#��Q��)\�H�"��� ���P�  l��p%�R(Dף�@��A�(�@\�?�Q��q=z�\�r�������HaD
�c?��@33Aף�@33�?ff���GA���9��z���HD�g@�Q�@�z,A�MAq="Aף�@�z$@��?�Q8?�qD�z�@�(AffAR�vA��A�zlA\�AR��@��@RhDffv@
�A��AA  PA�(�A�z�A=
�A�(XA��0A�D\���������@��U@��Aff:A�Affv@�ZD���?)\����)\�>��y@�G�@��0A��QAR�&A
�D
���(\��Q(��p�������?R�@  Aף,A�9DR����Q�q=��
ׅ�{N�
��
ד�R�n�\�"@�hDףP����z<��G%�������  8��(������1D�z�@�Qx@���z���(L���M�  �����p=��RD�Q�@ff:Aq=A�@�����?�p������>HQDR�~@
�A{zA��EA��@ffv@���@���\���YD{A��AA���A{�A  �AףdA�?AH�VA�Gq@HQD{�    R�~@
�A{zA��EA��@ffv@���@��DH����c�H���=
���@ףA���@)\�=)\��D��@33�?)\��33�?�(�@q=Aq=�A�QXA���@ PDR�R��½��Q@�33���Q@�ף ���a�=
g@33�>��D��)@�Q(���Q����  ������z���Ga��Q�@��D�Q��
�#@��)��W�)\�R���)\�)\���Qx�=D��?\�?q=J@�( ��G1�����������(��R�D�z�H���Q�
׳��Q����8�����(�����R8D  �?��������ף��
׋��Q����$�����(�� �D
׫@
��@q=
��ǿ33ӿ  �?R�B����Q0���D
��?���@ffA�G!�����������!@q=*���Y�f�DH�Z�ffB�����������d�)\[���\�ff2�\���\�Dq=A�G��ף �=
7@��@�p��q=������G��׳DH�N�\�R���=
w��!����̌�  ��R���)�
D��y�ff���G��)\��z��������H���z� 0D=
'��z����������,�  &�����ף.�),	D�(|@  ������33®G��)\�q=� �����.DR���ף��)\g�ף��=
,��z�33<�{6� ��:D
�s��Q<�\����(��\�®G;�R�¤pK��QE��D33_�{������q=��H� �)\?�{s��L¸���RD\�A�G������z����E��z��R�¤pM�H�&��D�̴@��pA{�?��,�\�*�=
���G����
�6�)�D��@��$A�Aף�@H��?  ��
��  H������i
D)\/A��yA�(�A�p�A
ׅAR�NAR��@ffA�ſ�9	D  ��R��@��-A�(�A�p�A�?AR�A=
W>�̄@Rx	DH�z?�Gq�{�@��=A  �A�G�A)\OAffA�Q�?)�D��Aף A�G�@  �A�G�A�z�AH�B�(�A��A͜DH����	A��A���@�(|A)\�A\��A��Bq=�AHD���?q=�?q=&A��5A
��@ף�A��A��A33B��D33�@R�AH�A��A�A�cA�p�AR��A��BH�D33��  `@q=�@\��@q=^A��mA��!Aף�A���A)D=
�>�k�H�z@��@  �@��dAףtAף(A  �A��
D
��q=��R�>�q=��   �)\?�q=�@���@33@)�	D�Q��  H��GA��p���G	��Q��  ����?\�2@{�	D�>�(����E�33?�ff��33��(��
����Q�?��D��@�G�@��@R����G��q=�\���Ga���5��RD
�C@�A��AH�@��9�R��\���\��?��5@ �D\�B�
ף<���@���@33@{��ף������G�)D��33����
ד@  �@H��>  ��\���H�6�)�
D  ��
�C�33��\�B��W@  `@\�B�  ��G	���D��������9�ffj��p9��¥��������ff��� D�Q�H�*�H�>�
�[�q=���[���������=
7���D�(,@)\?���
����0��pa��z0�
ד������D�(\�  @��Q8�H�6�H�J�
�g�q=���g��� ��xD�Q@���ף�?  @������(���E�ffv��pE���D��?{n@)\�>{>@=
W?R���)\��Q,���\��N	D  �?��U@=
�@��@=
�@��%@R���R����Q���D  ��  �>��?{~@�?{N@��?R���)\�E	D�̬?������?�(L@q=�@ף�?q=�@�(@�����DH���G��G�\�������������������åD���
�'�q=�q=*�q=������p����	��p��>D)\Ͽ�+���A��(,��(D��((�R���G���#��D��@)\@�;@  ���̤������̜�
�S�=
��D  h�q=��((��9�  ��33��33��33���z����D��)@��=�������R���̒�  ��  ��  ��
D�(�@�z A=
���(\�R��\�b��O���9���Q��D���(,�
�#�ףh�H����(���9��Q�����
Dq=�@  ��ף�?���@�'����)\���G�����`D��)�������i���q=��\���)\w�ף������� D�p���(��33C��(��q=n�
�C�����=
��  ��{�C�p���̺����=
����®G�)\���1�q=9�CףAq=���p��G��R����G��q=��=
���®� DR�A��A�(���z�����R�V����H��)\W��D�p�@�pYA=
�Aff�@�(���A�   �\���=
+�3D�(DA�p�A���A\�B��Aף,A\�B>�Q�@
�s�H�D=
�@
וA33�A�GB�p"B�p�A{�A��@�+A�;D���(�@��A�z�A
��A{BR��AR�vAq=�@HaDH�@  0@�A
׫A33�A�GB�p-B�p�A{�A�SDR���\��>Hế�Q�@�(�A��AH��A��B�«A��D������33ӿ)\_�R�>@
�sA�G�Aף�A�zB=�D
ד�������-�ף�������ѿ��)A�Q�A��A3cD{��)\���Q���?��(����  0��(AH�nAq�D�p!�333��}��̎��z�����z���pM��z��Q�C�M��G���(�����)\�����Q��� ®G��3��CR�����d�33��{��=
��ף ��z�q=���z�{t�C�G�?q=�>��H����  ������33���p
��(��)\�C{�����p�����ף�����q=
�)\�33#®��Cq=j�����p��G�33�����ff�H��  !�f��C��@q=�?���ffV�H���G��  ��H������
��C�(�@��AR��@H�z��G?��5��zX�����
���H�C�����p@ff�@\�R@��9��둿�Q(�33w��Q��=:�Cq=�@�̐@�z
A�QVA��AH�b@�©@�(t@{�R��C
���½@�Q�@q=A{NA�A��A@�G�@33S@�D33'A��A=
�A)\gAR��Aף�A)\�A�WA
�sA3SD�G�AH��A���A�(B���A  B�� B�QB��A�|D�̔@�z�A=
 B���A��B{B�� B\�3B��$B.D)\���z�ף�Aq=�A��A
�B�Q�A�BףB)LD�zx�{��H����L?  4A��+A�p�A�(tA��A� D�(��Q���(������R���)\@���?R�A  �@�N D��L?R����������\������\�R@��1@�A3cD\��A���AH�A�zT?�̌�  �>�G�AH��A���A�D33���A�(�A�(0Aף��  �33����<A{�A3SD�̌?  ��\��A���A��AAH�Z�ff
�  ��\�NAf�Dff��33��337���MAR�ZA��@��)���}�333�uDR�n@{^���=
���̄A33�Aq=
A�z����A� xD�p�@���@33?ff�?33k��(�A\��A��JA{6��QD=
������K@ף��H�:��Q�ff�A�̆A�pA=
D{.A��@�G%A��`A��@R��@=
'@�p�A
��A �Dq=
��A  �@R�Aff>Aff�@���@ff�>�(�A��D�( A33�@
ׅA�(BA�p�A�G�A)\KA��\A)\A=�	D�̤@\�RA  0A=
�A�G�Aף�A�z�AH�A��A�@D����  ���Q�@33�@�{A�(2AH�rA�G�A)\;A��D=
��(�=
W��̔@)\@��UAffA�MAff�A͜D33s@�Q�?R����G�>33A�G�@)\�A33IA���A3sD�̔���Y��pM����R���33s@���?�QHA���@ �D����337������( ��G���(0�  @����=
�@��D��i@��I�����  ������(H����ff&?�G���DH�
A�GEA���@�Ga?�@�z@�u����?�GA��D)\?A��A�Q�A��A�pMA��A�zdA{A�zTAR�D��U@��tA
׿A=
�Aף�A�p�A
ןA���A�7AH�Dq=FA�{A��A�zB{ BH�B\��A�zB{�A�D��q@�Q�A=
�AR��A{� B./B��BHaB{�B�D{�
�S@33{A�Q�A  �AR�B�Q-B�B=
�A�[DR�F@��(@�G�@�p�A�(�A�kB�#+Bq�9B=�B�9D\���­�ף�����Q$A��YA\��A  	B��Bf�D���?R���)\��q=���k��7A��lA�(�A��Bf�D  d���P����
ד�\���R�r���1�)\?�QHA)�
D����G������������
׸�ף��{��fff��JD)\�@���?H�N��;�����G��  ����]�Hế @
DR���(,�33���̨�33���z��ף��)\���(���hDq=
Aף�>ff�@��?)\G��(4�)\�����q=��
gD������@
�c����?)\����ףt������¥�=�D��Aף�@\�RA��@�'A���@{�����fft���D)\����@��@R�*A   @)\�@�zt@H�&����<D
ף?HᚿH��@
�S@33?A��q@�(A33�@ff��WD=
�>���?q=J��Q�@R�n@��EAff�@H�Aף�@�bDR�BA�pIA��]A{6A�p�Aff~A�Q�A���A�̮A �Dף(�ף�?33@�U@=
W?q=A��@  `A\��@��D=
W�ff^��pݿ��������G!����@   @q=*A3cDH���33����q=������
�����	���u�H���HD�pe@�Q ����q=������(t�q="��̠���a@ͬ
D����ff6�=
���D�R���R�*�  $���)\7�\_	DR����Q:��� �ffb�{��ff��{~�)\w�H�b��SDH��q=V��������ף���������z�������D��(@�G��  ,��z����Y����ff��)\�)\��
7D��h?33c@�(���p�33��=
K�q=�����R���5	DR�~@�z�@���@ff&����R�D�)\���l��G��H	D��Q�q=J@q=�@R��@�(������
�Q��z���y��"	D�?�����k@H�@)\�@33s�����I��(��(	D���=R�?��L���q@{�@\��@��Y�������G���D����(�
�����\����p��
�ÿ)\�R�n��BD�G�?33��  ��)\�������zt�q=:�q=��33�H�DR�@)\o@
ף�ף��  ��q=������(ܾ�p@\/D�z,�����G��ff~���|��zt��̀���A�)\3��!Dף<A�G�?)\_@  �@����Q��)\_��������3�D�pͿ��"A�Q��G�?�GI@H�����=
���G���K	D�(�@�̔@��A��@q=AffAq=
?
�#?�z�?q]D�zA\�hAH�NA���A=
_A)\�A�p�A�AR�A=jD���A
��AH��A=
�A�#B��A�zB�B�(�A��D�G��\�
A��A\��AR��A�B�̴Aף�AR��A�D  �ף���@=
sA\��AR��A=
�A�̦Aף�A�XDR�>�)\���QD��{@)\CAR��AH�A33�A���AH1	D��I���y��̊����=
���Ѿ=
�@��@q=�A=�	D��@�#�)\S�)\o�  �������p�?��A�(�@�5
D=
�?q=�@����z8��zT�\���
ד�q=j@ףA��DR��
��33����������������ףf��Q���DH���{��ף��=
[��z���Q���Q���Q�=
����DR��@ף����,����)\���̚�ף��ף�������<	D��@��]A���@��x��p��Q8>=
G�R�v�)\���qDH�J�)\o@33+A=
/@�����(��)\?���y�R�����D�����#�
�S���i@{����a�H�F�ף ��G��\�D\�B?�G��)\�33#���@�c���U�R�:��z��L	D)\A�'AH�Z@  �>��@��aA���@��h�H���\�D�p-�  �@�Q�@��5?�p�ff�@\�6A�z\@33���D�Q8>��!����@{�@
�c?����(�@�p9A  h@�Q	D=
'@\�2@
ף=ףA��(A  `@���>��@33cA� 
DR�N@H�@ף�@
�S@�QPA�z\A)\�@
�c@�'A�3D���@���@q="A�%A��@\��Aף�A�z0A��@H�
D\��ף @��@33�@�z Aq=�@�zpAף|A
�A)�D\������q=������?333?ף �ff�@)\A �
D���@q=��
�#�R��?=
�@\��@�Q�@���@�(lA��
D)\����@{���Q(����?�̜@�Q�@{�@)\�@{TD���H��)\�33�
�7�{���p}��̬���qM	D��x@=
���G���G@���33��33S�)\��\�"@�|
D��@{
A�k������Q�@���=
7��Q�?�p�@��
D��?\��@�A{�?��?��A�E?\�¿��1@)�	Dfff�ף�R�@
��@)\�
�#�  �@�5�
ף��a	Dq=
��Q���p��
ף>R��@�̔�=
����@�����D����)\���<��'�)\����33+��Q,��pm�=�D����
����z�{R�ף<�����q=J��Q@��pA�f&
D=
'A��A�zD@��h?�(,��̬���X@���@q=ʿ��	D
�#��(�@���@\�?33ӿ  ��q=z��zT?=
�@�D�z$��(����@)\@
��=
��q=��)\��R�޿ �D=
���G�����=
W?��������=
�ףD�33/��7D)\�?ff&��p��)\���p-@��?���q=��R�&�f�D�p�?fff@R�^��(\�  ��{�@�G@�;�H���D�GA��4AH�RA)\A�z�@�e@�Q`A33KA���@HaD��?R�.AffJA�QhA�� A)\�@�p�@��uAף`A��Dq=
@  `@�GQA��lA�p�A)\CAq=A\��@�(�A��D�pm@
׻@R��@�Q�A�(�A��AR�~A��UAף,A{�Dq=��{��ff�?��@�?A33[A�yA��1A�zA�|DR��{���pݿ�(�>\��?��5A�GQA33oA�'A�C
D�z���Q���G%�
���R����G�R��@=
A�� A PD��<�����\�������p���(��H�n�33���g��DHᚿ�QP��G��q=���̺����
׋��������3�DףP@33@�(�ffj��Qt�R���=
���c�{N���D�G�@���@H��@=
����)��3�ff���pE�H�"��gD�(ܿ�z@\��@
׋@{���GE�33O��(����`��	D��@�(,@)\�@
�A�zA��A��p��ף���U��lD�G�H�z��z����̿�z�?ff�>��5�  �������)D��<@�(���Qx��(,��̬?��@��Y@\����T�
D)\7A\�fA\��@
�'A�QA��LA\��A��mA33C@��D��yAף�Aq=�A���AH��A��A�p�A��A
��A�D33GA\��A�B��)B��Bq=B)\B�#B\�0Bq�D�Ga��CA���Aq=B=
)BR�B)\B�zBף"B�qD\��A�̌Aff�A�6B�dB�QpB  OBף`B��YB��.D��B�Q�BH��B���B��CR�C��C
�C  C�5-D������B���B�Q�BR8�B�C�0C�#C\�Cf�-Dף @ף���#�B���B
W�Bq=�Bq=C3�Cf�Cf�-D  ��ף@ף��ף�B�G�B
��Bq��Bq�
C3sC��(D��������H�����f�B=��B��B  �Bq=�BH�#D����Q!��Q"®G�ff3���SB���B.�B{�BH�$D  `@����Q��Q®G
�ff%���aB���B.�B=�$D��?�z�@�q�����R��
���gBff�B�[%D\��?q=J@��@��T����®G�����nB�#&D�QH@�̜@�G�@ףAR�"��Q���Q��q=��q=� `$D�����{�q=
��E�R�.@
׉�ff�ff�)\�.&D=
�@
�#>\�R@��@ff�@33A�( �=
��=
���Q%D�(\���q@��Q����)\�?ף@@�Q�@33W��GR(D�(@A�	Aף|A�A��=A{ZA�QpA�(�A�Q����'D��H���A�­@ffJAH�@�A
�'A{>A{vA�&D�Q��ff�=
G@�����z�@{.��p=@)\�@
��@��%D  ���Q��ff"�{�?q=ʿ�z�@�µ�H��?R�^@3�,D�G�A�G�A33�A{�A�(�Aף�A33B���A���A�,D�?ף�Aף�A\��A�p�A��A  �AH�B�G�A��+DR���\�r����A���AH�A�SA
��A�Q�A{�A�W,D=
7@ff�{n�
��A
��A�Aף�AR��A33�A
�-D��@33�@��(@
�s@���A���A��A\��A�QB�.,D�(��
�#�{@)\/���ȿR��AR��Aף�A=
wA��-D��@)\���p�@���@�z$@)\o@33�A33�A��A��)D��h�  �{j�q=�����
�?��-�R��A�pqA�!D���=
H���2��QH�)\5�)�=�{9�H��
�D)\���zu�)ܗ�q=��  ������̈�R���Ha��D�z���z�����#����®G���̙�{��  ��\�$D{=B�&B)\{A���33	����z	�=
���(��q�&D�A��]BffGB�p�A�Q8�ף���(��33���G��)�$D�Q���L�H�<B�Q&B\�zA�Q��ff	��Q���	�{�!D��=����R�>�ffB��A\�r@�G��H�8�ף#��cD{��=
����¤p��R��A337A�g��G�����=:!D��uAq=:��zl�ff���Gm���Bff�A�Ga?�G
�3CD�����@�Q,�����G�����ף�A��A)\��q=D�p�33����H��
���  �q=����A��-AR� DR�nA\��@
׃��eA�(|���|�ף����}��G�A=� Dף�33gA��@  ����]A���q=��ff��ף���"D�p�@ff�@���A�z`A�p�@�(�AףP@��	�=
��3�#D=
�@q=>AR�6AR��A  �Aq=&A���A)\�@�z��#D�5���?��A�p	A{�AR�vA���@�G�A�̔@{D&D��IA�QA
�_A�G�A��A�pB�(�A�G�A=
BHa&Dff�>��PA�#A=
gAH�A��Aq=B���AH�A��'D���@33�@���A���AR��A{�A�Q�A
�*B�z
B�:%D  ,�33���̄�33A
׳@�pA{�A�Q�A��A��&D���@{��H�?�z�?�(hAH�:Aff~A\��A�̸A�L#D�z\�=
�����E���=�H�:?ff��@ףA�R#D�Q�==
[��(��=
���C��z<���Q?ף ��p@=�#D)\�?H��?�A��Q��{����)�\�"��(@�Ǿ�	 D�(l�q=R���P�ף���(���(������)\���E� Dff~��G���Q��������*����=�{%®G#�
�D��ѿ�Q��33¤p��R����z1�q=�q=D�ף+��LD�piA33OA��<����R���  ��q=���½�H�	� � D���@��A  �A���?\�N�ף4�333�
���)\��\?$D
�oA�Q�A�	B��BR��A�@��l@\�r@
����'D=
kA�p�A��	B�GDBR�=Bq=�A�(�A��A
דA��'D
�#�ffjA��A��	B�DB\�=B���A
ׅA�̒A �'D��ѿ=
׿�(PA  �A33B\�=B  7B���A�pqA �&D  @��z�����( A  �Aff�A\�1B  +B���A\�'D��@R��?��̾�G�  dA���A�(B�BB��;B�)D{�@H�A���@�G�@  �@��AR�	B��B�GWB��*D��@H�>A)\�AR�RA�z8A
�7A�p�A�$BH�7B=*Dף0�R�~@R�A\�VA\�&A�QA�A)\�AףBHA-D��IA��AR��Aq=�A�(�A�(�A=
�AR��A�B{�-D33@\�nAffBA��Aף�A\��A\��A�p�A��A��.D�(�@���@�Q�Aq=�A�(�A��A��B���A�z�A  *D�����u��QP���ѾH�J��zd@�(A  PA   A3�'D33�33���(���©���	���5��(��)\?���@�H*D�pA��?�z��H�b�{>�H�:?���R��@ffA�3D��B�>B�QB=
�A{�A�z�A�BףB��+B��6Dף<A��GB�GmB�zLB)\�A33BffB
�JB��?B{47D���?
�WAR�NB{tB�GSB�zB  B33BףQB�:8D33�@���@R��A�_Bq=�B�cBH�Bff&B��/Bf6:D���@�z@A�[A�(�A
�~B��B3��B��3B�FB�X8DR���ף�>q=�@ף�@�z�A  aB.�B\�eB��B\�:D��%A��8@�-AR�nA���A�G�A33�BH�B�z�B{4;Dq=�?H�6A{~@ff>A  �A���A���A)\�B=
�B��<D��@q=�@)\�A)\'A��A��A��A��B
W�B�*>D��@��=AH�NAq=�A�}A  �A���Aff�A)\%Bff>D{n?�G�@�zLA��]A��A  �A�p�Aq=�A
��AH�<D\����̼�q=
�ff�@���@=
�AR�A�̎A���A��=D)\_@��U�q=���<@=
A�Q(A���A\�VAR��A P?D��@�'A��i@\��@=
A�p�A{�AH��A33�A�!=D���pݿ�G�?q=���z���(�?R��@ףA��A{9D����p�����33k�q=���̤�
�s�  ��p���;D��%A����Gm�H���33��H�2�  $��z�����?�~:D{����@��(��(���zD�ף���y�=
k��G�� >D�z`A�pA��AR�^@���  �?��@�˿��(�)�=D)\��)\_A�QA���Aq=Z@���=
�?�p�@�zԿ @BD�z�A��A�(�Aף�AR�B�£A  <A��A
ױA�,BD����{�A��A���Aq=�A�B)\�A337A��ARh@Dq=��
���=
A��A33�A��A�z�A��QA�(�@�P@D�p��{������A  Aq=�AR��A��A�KA�@D�p-@��@)\�������z@A)\?A���Aff�A33�Af�>D������������]�ffb�q=:@��5@���AH�FA��CD��A�=A�zhA\�bAH��@�G�@�̾Aq=�Aq=BfvBD{��  pA�(�@�p	A�A33�?��Y?�G�AR��A�@Dff��py�33�@�Gq����=
��  �����(Aq�BD�(0A{�?\���H�A��@33A�GAף @�p@\?AD=
���G�@�����,�q="A\��?R�n@=
W@�pm���AD��	@q=��{�@�p-�ff
�ףDAH�J@�(�@�Q�@3�>D�pI�=
'��G���̴���t��������R��R����>Dף ���q�33O�)\��\���z��  ��
�3�H�>��;D���{B�����\����Q��q=���p���z�H�F���>D��AA   @
�#���I�33'�)\�������t�  ���>D���R�A��L=�p���p�ffN��������{���%BD  �A��`A)\�Aff�Aף`A���?fff@�'�q=Au@D�Q��
�A�G�@�G�AףA���@���q=J�{�f6=D�O����)\O�{�����@�(L�ff���Q�����33>D��|@�z�ף|���5?R�޿{&A\�B?  ࿤pe�R�?D\��@�z$A��,�)\��G�@H�@��AH��@\��@� @Dף�?R��@\�:A�����G�R�A=
�@R��A�A�u?DH�*�����G�@
�A)\�  ,�  �@33S@R�vA\�CD33�A�[A��qA��A��A\�FA�̴@33�A���Aq}FD�;A���A���Aף�AףB�pB=
�A���A�zB3�EDq=j��� A��A�Q�A)\�A  �A��B�£A)\[A��BDףD�33�)\��R�BA  A{.A��A�G�AH�A)�=D\���H���{
�ff��������{��{����%@�a>D��@
׃��(��R��������)\��33��H�:?�=D����Gὤp�������
®G���Q��H������ @9Dף��q=�����=
�33E�
�S���$�R���{��=�6D�p!�)\������q=��ff<�\�m�33|��QM�R��{D7Dq=
@����{�����������3�d�\�s��D��U:D�QDAH�fAH�@
�_������a���
�3��zB�{�3D�(��  X��p5��p��=
�
�'��z�i��u����6D�=A33_�=
׿���>���������ff���z:�\�1D33���G�ff�ף��)\��{��)\C��(L���C��3D�p	A��D�H���{��
�_��G=�)\��  !���)�ף3D���A�GM��G��q=���(h���E����{#��5D��AH��@�p�A=
��33�@)\����������Ge�q]:D�Q�A33�A=
�AH�B�aA��A���=q=FA��hA�g7D�p=�ff�@��pAףhA=
�AR�@��`A�;���?�r7D{.>R�:�
��@�sA)\kAff�A��@�cA��8�e3D����Q��=
���p�H�z�  ��H��@��\�R���3;D���A�(hAH�jA��5@=
�A���A���Aq=B�G�AfV>D��PA{/B�z�A
��Aq=~AR�
B�(+B{)B�pKBH�@DR�A�³A��TB��B��B�z�Aff0B
�PB��NB=:CDq="A�z�A�pB�Q}B�z<B�(=B��B��XBffyBRX;Dq=������?�q=�?ff�A�pyA�(|AH�z@��A�6D�̚��K���"�\����(��33GAףp���e�H�v���/D�(���z6��L����)\f��(2¸]�q=��H���=
0D���>����H�4� ���R8��d�\�0�R�V�=
��HA/D��H�)\/�{���pA®ǟ� ����Qq¸=��z��f�0D\��@�(,@��E@�p���*¸����  Z���%�
�0DH���H�@��@ff&@)\��{,���р���[���0D�?
�#=
׻@R�.@�QH@�����)�=
�����h3D  ,A�z4Aף,A���A�WA{^Aq=F�����{}��6D�GeAף�AH��A���A���A�z�A��A�Q�?�G���`3D)\g�����)Aff2A\�*A��A��UA  \A�QH�
�4D33�@���=
�@�sA  |A�(tAR��A���A�̒A3C6D{�@ף8AH�:�\�6A�G�A��A���Aq=�A��A�R3D�(<�q=���Ga�H�j�33��ff&AH�.A=
'A�(�A�94D=
g@ff�Hᚿ��X@�1�ףP@�(`AףhA��`A�G6D�A�G=A)\�=�Q�@��9Aff6��7A
ױA{�A
75D�Q���p}@q=�@{��  0@33�@���=
�@�AN3D�z��ff>��k�)\���G=��z���z��  l�=
׾�9D���A�z�A��PA�(�A=
�A��QA�z�A�G�A33#A��8D�(,�{�A��iA��%Aף�A��AH�&A���A�¯A��6D�z���G)���eA)\�@{@=
+A��dA\�"@�A�6D)\���(�33?�  PA��@�̌?�AH�NA�?332D)\���Q���p�������p����@�\����������a2DH�:?����z����������(l��G5��py�
���=Z5D{>A��IA��������a�{��=
A��?�pm��
7D�Q�@��A���A���?�z?����   �33oA���@�7:D33KA��AR��A�G BffjA�zTA�z�@��,@33�A\�;D
׫@\��Aף�A
�B��B�(�A33�A�(,A�A{�;D���>ff�@33�A�G�A�(B{B�̢A
חA�p1A
W=D�G�@
��@
�GA��A���A�Q/Bq=2B��A�(�A��=D)\�?��@
�A��YA�z�A�GB��3BR�6B{�A=�<D�zD�����H�@�p�@ף(A��A  �A�'B�p*B=�>D  �@�k@���@�p9AR�>A�Q�A���A  B�BB3>D=
��z�@���?�(,@�A��AH�rA=
�A\�
B�c=D)\�33����	@�k���L>��@q=�@=
KA��Aõ<D{.�R���q=�������h��G!�ף�@33�@�A�Q<D�ǿ����ף��{��Q�ff��\����p-@\�B@�x?D��IA��0A�GAH�@R�n@�'A��@�zA�uA�@D��@��pA  XA�z,AףA���@H�NA��A�/A��>D{��)\?���A�� AH�@ff6@�p=?��@�p�@�j@Dף�@q=�?��q@��A�GmA��AA��A�Q�@�(dA�G?D����{@{N��E��p=A�z$A���@q=�@�p=@{�=D������1�\����Q�q=���G�@R�n@�G�?�p��{@D  A��<@��̿�p�@q=���@ףlA�SA�((A�>Dff��ff�?  �������1���ף��H��@���@�=Dff&���	�fff?�̴�33#��[�����p��{�@�b=D=
���%��z(�
׃�\���{B������,����{?D���@33�@ff�@  p�  �@��L�33���µ?ף��=�@DH��@��eA=
GAף<A��u@�pUA�G�@)\@�(A��ADR�^@�1A�̎AR�~A�QtAq=�@\��A�Q$A=
�@  CD��@�pAH�zA��Aq=�A=
�AH�>A�p�A{nA��DD���@ףDA�Q|AH�A��A��A�z�AH�AH��A BDH�6�  p��(\?H�@H�>A��Aq=�A=
�AH�A�ADq=���(L�\���ף�ף@@��)A=
�A33wA��lA  CD\��@  p@����    ��@�pAH�zA��Aq=�AHq8D��(���ףG�(��z�\��ff���(��q�9D{�@�(�
���(�H�5��(�R����H���W5DR���ffF��Z�33F��K�q=y��Z�{H��(:�q]4Dq=z�  ���z���(j�
�U��([¤p���(j�R�W��C0D33���z�����
��H��R���Ha��q=��H��R�0D�z4@�GY�����Q	¸��q=��{��q�����=�2D���@��A����)\/�ff��H��.��=
r�)\w�)1D=
��)\?�QX@�QP��p��{�ף��������®�1D�;@\�B�)\_@���@�p!�  `�R���33���B��q�2D��U@ף�@���>\��@ffA  ��\�*�  ���z���K5D�'A��\A��A�Q,Aff�A���A{n@\�B�q=��RH6D��|@R�fA{�A��A�kA  �A\��A�p�@ףp@R�8D  $A33cA)\�A{�A��A���A  �A�G	B)\�A3�6D�G��µ?
׫@�p}A�p�AH�A��A)\�A���A{6DR����4������8@��UA���A=
�A\�ZA��A55D)\O�=
����h�����33����!A)\WA��AR�&A3�6D=
�@R�@  ���G��?
ף@�pyA�p�AH�A)4D��!��z���(����%����=
����)\�@�A�q4D��?�Q�33C��G���Q��̌�33����Y�q=�@��2D=
���(��
�o��Q��(L�
�s�\����]����qm4D���@)\��\��?�p	��G�����p�)\���p�� P3DR����Q�?�����(<���P�\����-���T�=
���A7D�z|A�5A�A  4A�pMAR�.@33AR��@R�@f�8Dq=�@�̦A��A�Q�A\��A�G�A���@�QTA�z A�;8D����y@�p�A�sA���AffrA��A�(�@��AAf�:DR�*A  A�iA���A��A�(B\��A�G�Aff�Af6;D  �?R�>A  ,A�}A���A��A�(B\��A�G�Aq;D�Ǿ�(\?�z8A��%AH�vA��A  �A��B�p�AH:D{��\����U�H��@�p�@
�3A�(�A�z�A��A
�7D\����I�
�O�
�;�����)\��5@H�AffbA׃6D����)\c�33���Q���Q��
���ף �{>���LA��6D
׃?ף��H�R�����{��{��H���Q���Q����9D�p=A��MAq=�@��������p���p��  �@\��@H�9D��ѾH�6A)\GA��@  �{��\���\���H�@)L7D�G�
��ff@�QH@H�*��G1��Qt�\�z�\�f�R�6D���q=*���0�q=J?���?R�~�q=F�ף���� P7D)\�?��u=�Q�H��q=
@�(L@=
'��Q0�)\s�)7D���R��>  ���G%�
�+��̌?�Q@H�j��GA� 5D{��  ��(��=
��(���p��H��������9� 0D  �����  ��=
�����{�R��R����z��R�0Dף�?�����z������  ���z��\��33����R81D  0@�(�@��u��z������  ���z��\��33��1D��?H�@=
�@�z\��­�q=���G���µ�33 ®�2D�̌@��@
�A��)A{�\���=
��{��\����L+D)\���(���p���p��ff��33���;�33@�R�:��H*D������ף����������H���p,�33L¤pP¤�,D��!A���@H�����������m�
�O����R�#�)�-D�{@��`A
�A�p���zt�=
[�=
/�����z����/D�pA�QDA��Aף�A�p5�{��33��ff&��Q8� �1D=
�@��pA��AH��Aff�A
ד��Ga�)\�?
׃@�80D���{�?33A{ZA  �A��A��\���)\�>1D\��@
ף�{�@�z\A��Aף�A�(�A�̼�  ����0DH�ڿ�@)\?�R�n@�AA  �A���A�z�A����q3D�Q(A��Aq=NA���@  dAR��A�(�A\�B�QB��5DR�A��A
בA�z�A���A)\�A=
 B��Bq=8B� 7D��@�kA  �A�Q�A���A{�A
��A�GB  %B�^6D��A���@33;A�±A{�AR��A
יA���A�(	B3�7Dq=�@\�R@��	A�(�A�Q�Aף�A�G�Aff�A{BR�7D{.����@�G@H�A�̎A���A�G�A���A=
�A�L<D\��A33�A�½A��A  �A�B��7B��0Bq=AB=Z<D=
W>q=�AH�A�p�A33�A��A�B��8B��1BE=DH�j@�Qx@���Aq=�A���A\��A=
�A33B�GGB=�<DH�*�  �?H�?q=�AH�A�p�A33�A��A�B��=D
כ@��@
׻@\��@33�A
׻Aff�A�(�A�Q BHa=D�G�=
G@�G�>��@q=�@��A�­A�Q�A{�A�i=D�>ףп)\O@��?��@ff�@�(�A�̮A)\�A~>Dq=�@ff�@�(,@���@�z�@��A�QAR��A)\�A�u?D�w@=
A�A���@H�6A�(AH�FAq=JA��A��ED  �A���A��B�GBq= BR�B=
	BR�B\�B)<HD��AffBH�B�(-B�-Bף&B�:B�p/B�>B�HD��Q��zA�	B��BH�)Bff*B)\#B
�6B�(,B�GD�����G�\��@�QB��B{%B��%B\�B=
2B PGD��տ�7��(l���@�G�A�BffB��BH�B��IDH�A�( A���@��@�pyA)\$B
�3B�EBףEB�NID��ٿ)\�@���@��@�G�@q=^A\�B=
-B�Q>B��ID  @��?�#A��A��@�G�@��A\�&B=
6B\JD�G�?ףP@��?
�3A�A��A���@33�A��*B�JD�z���Q8?{>@\��?33/A�zA�GA�Q�@H�A3�ID33���Q���(ܾ���?���=��A{A���@��@q}ID���)\���!�\�¿H�:?�Qx�)\A�G�@H�@)LHDף������Q�������G���G������(|@�G@�ZID)\�@q=
�ףп��1��zD�
��\�B>�G��R�A �HDH�*���?�pM�����ff�����)\��R������GDffF�ף���ſ������������p�\���\����<GD��,�����������(���!�  4�ף8��z(��%GD�Q��
�C�����G�33�����)\'���9�ff>�ffFD)\?�ffV�����ff��=�H�����E�33W���i�3�ED33��G���̴����337���a�q=�\�j�  |�NFD���?\�¾�W�R�n������z�33C�=
��
�K�{�FDff&@ף�@{@�E�ף���u��������
׫���ED��y�ff��R�?=
׿�p������)\���G-�  X�fvED�z�=
���W�����  p����33�����ffN���EDff�?���p��������=���{������  ��7GD=
�@ף�@ff�@ff�?��i@q=�@�GQ@)\�>
ף�HJDff2A���A)\�A�̀A33CA��lA�AR�fAH�6A\�ID������1AR��A��A\��AR�BA�QlA��Aq=fAõHD�̤��¥�=
�@=
7A
�OAR�.Aף�@��Aף8A�ED��e�{���Q��q=�H�:�)\���(\�H������{�GD�p%Aף��R��33����?�p�@�A���@  @@.ED��!���u>��a��(��ff��ff��+�ף����L�q�EDR��?����p�?{F�q=���z�����ףp��(?�NHD�Q,A�(HAq=@  LA{ο�Q���G����@�GA~GDףP��Q�@  A��Y�
�A
כ��Q ��� ��̌?��FD�%�H���@�p�@�[���@ff����I�{J���GD�z�@��?�����A��0A��u?��4A
�C�)\��AID  �@q=A���@33s@�iA�z�AR��@ff�A�(@��KDR�.A)\�A�z�A
׏A�kA���A
��A=
�A���Aq�KD�p}�H�AH�rA\��A��A�[A  �A���A��A��HD��D���T��Q��7@�Q�@�@�µ?=
CAH�^A��JD�GA)\��=
��ff�@33/A�puA�(LA  A�(�A׃JD�+���@�̜��z�����@�z$AR�jA�pAA�GA��HD�p���p�
�#��E���T����=
7@  �@�p�@�ID�µ?  ���p���z�?ff.�q=>��Qx����@R�
ARHD�k�ף�H����%��G��Gi��y��̔���?��HDq=@\�����>ף��{��)\>R�B�\�R�)\��NHD�ǿ��Y?�5��z���G�  ��µ��[��k��HD���?�z�>��,@H�z��G�>)\�������(�>{>�q-HD���������>ffV�=
������Q ��Q���KD)\WA�p1A=
OA{6Aף\A��!A�z8A��@�(\@{dJD33����A��@�pA���@=
A�Q�@���@H���H�JD��@�����0A=
Aף(A�Aq=6AR��@{AID{������p!��W@  �?ff6@��?��l@
ף<f&ID�?�p��=
������x@\��?�W@��?=
�@õID)\@ף0@��R�.�\����(�@ףp@��@���@\_MDffjA��A�G�A�AR�>Aq=�@q=�A�G�A{�A��ND33�@  �A��A{�A�uA�(�AR�NA=
�A{�ARhMDR���)\>ףlAq=�Aff�A��A��@AR��@)\�A�ND�@��A��z$@�A��A
ןAףDA
�gAq=A\�PDH�:A��aAff
A  dA33�A��A�G�A�¿A)\�A3#OD{����@�p�@H�?���@��A���A���Aq=�Af�MDff��q=Z�H���ף�>����(?�(tA  �A�(�AfVLD  ��333����R�������
���z���((A  LA��LD�p@\�"�
���p��  ���z�����=
���KA�yJD\��ff��33C�33��R���\�b��;�����p9��^ID�p���Ga���=�����\���{��ף��33��H���ED�z��
׫�����p��R��� ®G=�\��
��)�CD)\���Q������z�ף�ף¤p9�33V��z'�fFCD���q=��=
��ff��
��  �  $���=�\�Z��HD�p�AR��A��YA��\�
���q=���u�\����(����JD33A=
�A�Q�A�z�A  �@�@\���
׫���!��eID�z��
�c@���A33�A)\�A�G�=����_��(<��CDff��������)\O�33�{����H���{� �@Dף�)\� �q=�����=
'�)\������N<Dq=��\����zQ�=
k�q=C�=
���������=
Q�=�AD��Aq=Z@�(���p���G�����{�������(T���BDq=�@{�A�AR���H���   �ff��)\���p�=�DD���@  0A��B\�fA
׻@�p��\�����}���@�AD=
#��Q��)\O?  �A=
�@q=������=
��z��33DD�GA{��q=�@q=Aף�A��PA�Q�@�Q���p�� @DD��L>�zA�z��ף�@�pAq=�A  TAR��@R��� �BD  ��������A@���H�z���u@q=�A  �@����CD�p�?ף��q=���Q�@�½�  �>q=�@{�A�A�rDD��@�G�@q=J?�p}?�A�p����@{*A�GB)\ED��i@)\A=
'A{�@�z�@�WA��Q@)\A�zdAR�AD��X�\��33���G����R���Q���z$�33��?D��8�H������ff~�R�j�)\���£�q=:�ף��q?D�>R�6�
���ף���Q|�ףh��Q��R����(8��BD�(\Aq=^A��@�3�q=��ף ��G�����\����BD\���;A��=A���>�(T���������z4����EDףtA  TA{�A��A�pyA��@�µ@
�3A�GA��FD��l@��A���A��AR��A�Q�A)\�@{A=
oA�\JD�zdA
׏A{B
��A��3B�z4B�GB{�A�G�A��IDffƿ�KA�p�A���A�p�A��-B�G.B{ B��A��KD{�@�z�@)\�A���AףB�zB�KB=
LB
�B�xLD33#@
�A=
A�µA)\�A
�&B�BR�UBq=VB��KD)\�R��>  �@ff�@
ףA�p�AH�BR�B��LB�uJD����� �  ���Q�?��>R�jA���AףB���A  KDq=*@��I��z����5�33�@33C@ף�Aq=�A�GB��KD��,@��@ff��(,�)\����@  �@q=�A
׽A�KD  @�����=
@��\�{����H����@  0@q=�A�kJD�G!�ף���z4�
�#�=
��)\����
��?�k>3�IDף��̌�����ff��\��33���p!��G����̽�{JD�Q@  �>�G�ף���z$��Q�==
��R������{$ID���R�N�����(��{*�����ף���G1��U�R�HD�Qؿ�����p���������E����R����QL��FDH�&���A�
׋���u�
׉�  ��  ��ff������ JD��xA�(�@�(\@����=
W>��տ{��{�����H�ID��u��uA�z�@��L@q=
������z��������HAKD  �@�Q�@\��Aq="A33A��E@=
�@��U@��Q?qMND=
CA��A���A=
Bף�A��A�ztA�G�A�zxA��ND�(�?\�bA�G�A)\�A��
Bff�AH�A  �A=
�A��ND����L?
�OA��A  �Aq=B=
�A��Aף�A
wODffv@�+@�̔@R��AR��A�̮AףB
��A�Q�A�aOD�����Ga@ff@q=�@{�A{�A�(�A�QB33�A)�OD�z�?q=�?�@ףp@)\�@)\�A)\�A�p�A��Bq�NDR�^��z�������?)\�>  @=
gA��A���AHAOD��?����\��=
W�ף@@��?
�s@  �A  �A�ND�̔���E�q=����������ѿ
�3�=
W���5A�
OD33s@��Y���5?�G1�{���Qؿq=
@�p}?�p=@�SD{�A�z�A\�vA�A
�WAffnA�iA)\�A  �A�aTD33�@H�A�G�A{�A\��AR��A  �A)\�A�(�A͜VDR�A�Q`Aq=�A�QB�p�A���A{�A)\�AR��A�>VD�(��ff�@��HA�z�A�pB��A�(�A�Q�A���A�ASD33?�R�V�  ����?H�A�G�A{�A\��A�paA�MD��������z
����̤�H��q=
�{���Qh�)MD��h�R����(�=
�R�������)\�����G�\�KDff���p	��Q����&�
�,��(	����H�N�{���LD�GA@=
��q=���(��H�� ��(��)\��\��RMD)\�?�z�@��u���l�33��ff®G�33��ff���gMDR��?=
W@�(�@=
�?�p��G���p��Q®G���NOD��@��A�/A
�_AףA�̬@��|�  ������RxODff&?�(A  A��9Aq=jA=
A���@ffr�����׃OD�Q8>�zT?=
AH�A��<A�mA��A)\�@�o�f�PD�G�@=
�@
׳@�SA�kAR��AH�A\�jA�Q0Aq�QD��@ffA�GA�AR��Aף�A���A�½A�(�A�RD���?�¥@�Aff"A��,A�G�A33�A�(�A�Q�A{TQD�p-���ѿ{@�Q�@{�@�pA33{A��A�z�A�RD33�@��@�pM@q=�@��AAףDA=
OAff�A�Q�A3�SD�(�@�Aף�@H��@33;A��A)\�A\��A�p�AHQTD{.@33�@33?A
�A��$AR�fA��A��A�Q�A
�TD��@��@=
Aף`A�G5AffFA{�Aff�A
׫A �TD�Q���p�?ff�@�GAH�ZA�/Aף@A33�A��A��SD�(L�33c�Hế�G�?�z�@
�'A���@��A)\OA�jTD{�?q=���Qؿ���>�G@  �@��EAq=A)\+A�;SD����Q8�q=������H��)\Ͽף0@��@�̜@ 0VD�=A\��@=
A  �@�z�@)\�@33#A�GiA�p�AfvTD�����p�@�Q8>\�@33���G���z?33S@���@ETD�E��p���̄@=
�\��?�������\�B���!@H�SD�翸%�����@���q=
�R�n�H��   ���TD�Qx@�z@ff�?33��=
�@�p�?��U@��>=
W��hUD)\@
��@��@\�r@=
G�)\A{~@\��@��(@�uVDff�@{�@�)A�(A��@��?\�NAR�A�z A�YWD�(d@�z�@{$A�(bA33EAH�8A���@�̃A��;A%XD33K@��@=
/AH�VA�z�A  xA�kA\��@33�A=zXDq=�?�(�@�A�QDA�(lA��Aף�A�z�A\�A�YDR�@
�s@��@��(A  lA��A���A�z�A�Q�A �YD{�?H�@�p�@�	A\�BAH�A�̖A
׵A)\�A�ZD�@R�~@R��@�G�@�p/A�zhA
וA�©A���A=�XDR����������   @�U@�(�@�!A�QdA{�A��TD�(��
ױ�H��  ���Q|�=
g�q=4�ff��  `�3sTD{��H��\�������R���H���zl��9�ף ���QDff"�
�'�{��H����������{���p��=
����PD�Q��\�j�  p�{�¤p�  ��(�������QD�W@�둿ף4�{:�33���p���=
��33��R(RD�@��@H�z?R���(�q=����������{���wTD
�A��5A�kA�#A)\�=q=���Q��  ��=
���^TD�Ǿ��A�/A�peA�GA
ף���(��p�����3�SD=
׿�z��p�@ףA\�JAffA   ����H�� 0VD33Aף�@�(�@���A��AH�A�̈Aff�@��@{$UD��ף�@��E@��,@=
?A��`A�p�AR�NA�G1@=:TDq=j�H���{�?�����u��zAff&A�Q\A�(AFSD�(t�33���z:��-��Q���̘�H�@R��@�GAf�RD�?������'�ffj�ff���(��ף���(�?��e@�lRD��̾�GY�R�����-���p�33������R�����?R�QD���{�H���z�=
K������p�����
��q]RDq=�?��u�
�#�ףh�ff����1�ףt�H����Q ��RD
ף����>\�¿�����G����q=F�\���
���R�UD33oAR�ZA  tAH�VA�zPA\� A=
�@
�#@)\Ͽ\UD�둿��\A�zHA��aAףDAq=>A�QA\��@�µ?� VD�G!@ף�?ף�A��pA=
�A��lA\�fAף6A33�@WD�zd@H��@ff�@33�A���A���A=
�A
׏A��oAͼVDף���(@R��@�zt@�(�A��A\��A  �A�̆A��[D
ןA�̖A)\�A��Aff�A  BH�B33B��B�5^D\�A��A{�A�QBffB
�BףBB�=B
�CB�H^D���>)\#A��A�z�A�B��B=
B
�CBR�>B)�^D33@ff&@�(HA��BH��AR�B��Bq=B=
MBH!gD�QB�BR�B)\6B�#�BH�B�B\�B�ǊB{4iD��A�%BR�.B��/B\�WBq��B�z�B���B���B�EjDף�@�IA��6B��?B  ABףhB�G�B�B�(�B�mD��5A{zA�p�A=
dBq=mB�pnB=
�B  �Bq��B�nDף�@=
�A33�A���A�}B�(�B�B{��B=��B�mD{��q=�?=
SA��A{�A)\kB\�tB��uB3��B)�oD\�A{~@
�#A�̬A���A�B ��B��B3��Bq-oD�p�����@R��?�(A���A��A�� B=��B�#�BõnD)\�ffv���@���=�z�@  �A�(�A\��A��}B��nDff�?��Q�333���@�?{�@ff�A\��A���A��nD  @����>��ȿ33c���@=
�>{�@ff�A\��A�oD\��@�U@�(�@ף @�?=
A  �@�Q,A=
�A
'qD�z�@�A�A�QA���@�p�@�GeAq=A�G�A�kD���ff���O��[�R�J�ףh��(�������H� �mD�GA��a����z���z��H��R���=
��Ga>��nD�̄@�GA)\�q=��H���R���\�B�
���G��{qDR�A�YA33�Aq=
�33�@H�AH�A�A��@3�rD)\�@ff~Aff�A=
�A{�@�G5A\�vA\�jA)\{A
gtD���@ףXA��AH��A��B  PA��A�³A�­A
�wD  dA�z�A�Q�A
�B�p%B��FB  �A\�BH�BHQxD�z�?\�zA�µA���A�zB{+BffLB�G�A33B�
yD��9@��@�z�A���Aff B{&B�6B  XB�z�A��vD�G�����ף���A�Q�A�(�A��B)\B�3B�wD���?�(��)\���zd�H�*A��A���A\�B�(B�yD\��@�zA��L�ff6@�Q�@{�A\��A33 BH�%BR�zD�Q�@�pyA�Q�AR��@��)A�Q@A�(�A�QBq=B�+yDff��)\?q=A�pA\�?q=Z@q=�@\��A=
�A�ezD��@\��=
�@��TA  hA�p�@�A�A
׿A�{D�(,@33�@���>\�A
�A��A��A�(0AR�FA��rDף����z��=
�  ��)\����ff��33���spD���H�)¸��z®G(�q=	�
���q=���p	� �rD=
A\��>��R����p���������Q��R�����qD�z$�
��@�(�ff�ף�  ��������H���pD  ���%�ףп=
�ff0�ף%�  ���.�®�oD�ǿ���{>��(L�  8�ף6�H�+�q=�=
5��sD�̄AףpA�G�@=
�@\�VA33�@�z���������\?rD)\����%A��A��?�G�����@ף���(�ff�åtD��A�W@�A�G�A\�*A�pAq=�A�A����'tD�(���(�@33�?  �A��A=
A
��@��lA  �@ÅvD�A  �@�̈A��-A���A�G�A�G�A�pyAq=�A�sD  `�����  ��ffF@�QH��WA\�>A��@��?�mD  ��  �q=��  ��33��=
���z(��pA�R����@pDR�NA�G1�ף����y�ף��)\��)\c���@�zT?{tjD����Q$�{	�{A�33�{#�)\�����ff���khDq=�ף���G��ף)�ףa�;�ףC�q=��(6�q�eD�#�H��33&�=
���RB��ףd��l¸F¤iD��LA�%@���  ���G}��Q��QW¤p1��Q9�)�gDq=���AR���{"��G�33����1�i�R�C��3kD��QA��A�̪A{2A)\?@��������q=���5� �mD=
#A�z�A��A�Q�A\��AH�RA�( �q=:@R����anD��!@�KAR��A�(�A�GB�̾A)\{A)\��{�@=�qD{bA�G�A���AH�B��B��@B��BR��A���@�-rD)\�?  sAq��A�B�A)$B{�B�EBf&B.�A=:tD{A  A=
�A�G�AffBH�DB��2B��eB��<Bq�sD33s����@��A�p�A��A��B{AB��.B  bBͼvD
�/Aף A)ܑA�Q�A�B��B\�8B=
mB��ZBH1xDq=�@�z�A��}A�k�AH��A��B{'B
�OB�(�B)xD�z���@
ׁA�ztA�ǻAq=�AףB��$B�MB�uxD33�?���?�z�@=
�A�p�A���A�p�Aq=!B)\+BדyD=
�@
��@�G�@��5A�̲A33�Aq��A33�A�3B�ArDq=���z���G�����)\�������(��  �>)\�?�SqD{n�  �q=��=
��������ff*���9�{Z�� nD��L��(��337��Q%�R��=
!�	���33���oDq=�@)\��33+����=
¤p�	�����=
�� `lD�GM��Q���z��q=��q=S�)\A�:�{=���%�ףlD��?�Q<�ff��  ���³�  O¸=��6�
�8���lDq=�?��@=
+�
כ�)\������J���8�332�3kDH����Q��ff��q=��)\C�{��
���=
h��(VrlD��@��쿸E��z�>ףH�=
���(�����{R� plD
�#�ff�@���)\O�  �>�GI��Q���z��q=��)lD�ǿ��̿��x@��\������q=b�������kD����p��R����Ga�����)\���p��  ��H�F���fD����̤��G���������ff���·��G����
�ÅhD  �@
����a�\�z�33{�)\#�ff����\�v���jD�pA�puA33���Q��q=�����)\��)\��(���jD��Y���@
�gA   �����p��R���{����HlD{�@H�@H�bA�p�A{�@
ף=�p��\�¿{~@=�gDH��R�&��Q4���q=�@R�F�q=��R���=
����gD�p��R���ff>�  L�q=z���%@ff^�{��\����7fD����G��33������z����{.����\����kDq=�A�cA
�KA�w��z@�(�?��$A�z�A
ף>�kD�@��A33�AR�nA�Qؿ  �@��i@
�GA��A�8jD{��ף`��(�A)\+A�A�(���Q��\�����@ �hD�z���GA�ff�{Aq=�@��5@�Q\�\���H��RhD���(�33g��QD��Q�@���?ף�>���33� �iD
��@  �@�둿\����̔�{nA�A�pA�Q�R�kD�(�@  xA{RA��@ff�?R�N@{�A���A�A\�nD��9A��AH��A���A�̒A\�JA�pmA�z	B�z�A\�mD  P���A��AH�A��A��qA\�A�p9A���A�^kD�((��(\���	�)\�@��UA�/AH�@�̌�q=�?H�kDq=
?���S�{οף�@q=^A�Q8A�(�@)\��xmD
��@\�Aff�33���Q�@q=bA{�A��A  PA=ZkD���(�)\���G)��G]�{���@�zTA\�.A��kDף�?33��\��?��?33�33?�{���G�@\�rA��lD=
W@��@)\O��(�@�p�@H���p	��GA@ff.A)�mD\�r@���@�zA��?R�A)\Affƿ�������@RkD��$�ף��q=J�
ף��(�����̬���=���q���iDH��ffr���5�   �
�����i�)\��{������ phD�G��{&����33��ףX�\�:�����QD��;�
7jD��@���?�GQ��GY�ף����������zP������lDH�*A�Q�A  DA��@��9�
�c?  �@�(�@ff�3�lD
�#<=
+Aff�A�(DA�p�@��8�fff?�Q�@�z�@��kD�G���������@�(XA=
�@�QH@������X������iD���QT��(T��%����@�G��33���G����E��[jD��I@\�����!���!���?���@��@��,��(P���oD�³A���A��A��EA��EA�Q�A33�AH��A�(�A�3qD��@=
�Aq=�Aff�A{�A�(�A���Aq=B�(�AnsD\�A�]A�(B��B��A)\�A�p�A�pBH�/B�qD  ��q=*@q=�@�Q�A��B��A)\�A�p�AH��AsD��@��ѿף�@H�BA��
B33B\��Aq=�A�Q�A�qD����ף0�{���(�?�(�@���A  B�(�A
יA)LvD�G�A��QA�A�7A=
�A�Q�A=
?BףKBR�'B�hxD33AH��A�z�A)\�A)\�Aף�A��B
�`B�pmB�xDq=��
��@q=�A
סAR��AR��A  �AףB�[B3�vD�Q��H��=
'@�(�A�{Aף�A�GaA��A33�A�i{D�̎A�pUA�(@A��A�zB�GBR�B�p�A)\#B�i{D    �̎A�pUA�(@A��A�zB�GBR�B�p�A��{D{@{@\��A��xA�cA�p�A)\$B�(B��!BfV|D�p�?��l@��l@ff�A�Q�A)\{A�G�A�G*B{B �|D33�?�QH@33�@33�@���A��AH�A�z�AH�0B\/}DR��?��X@
כ@H��@H��@��A�p�A�̘Aff�A�uzDff.�\���Q������
�s�
�s�ף`A�zA33A{D{DR�N@�p���½�����333��z��z��(�A�(LA�bvDq=��ff����������z��ף��H��H��ף�\wD��<@ף����U�  ��{��H��=
���G���G��q�wD{>@�p�@��Y�{&�q=���Q������G��=
c���wD
�#��;@�(�@ffZ�R�&�\���ף���p������q5�D  )B
�(BR�4B�@B���Aף�A�p�A)\�A\��A)$�Dq=
�
�&B�&B\�2B)\>B�z�A�Q�A��A=
�AH!�D�Q���G!��z&B�Q&B332B  >B���A���Aff�A3S�D��A�AH�AR�LB\�LB�pXBq=dB�B=
#B
_�D�p�>H�A�pA��A33NB=
NB��YBR�eB��B�?�Dף�@�z�@���AH�A\��A�GjB�jB  vBf�B�U�D�����z��
ף=q=A��A�(A=
MBH�LB��XB{��D)\A��@R�AףA�̚A{�A�AH�sBR�sB
��D�p���G�@�G�  �@
ף@H�jA�piA��`A33aB̈́�D�zd�
���(�?H��=
�?ff�?��1A�Q0A�'Af&�D��<�ף��=
3��p��ף�\��33��\�A�A{4�D�G�>ף �\���  ,�������q=����u���	A��D)\���G�������4���ף���GY�����{��
�D��@A\��@ף�@q=�@  @?H�z��G�@\�¿  �@{|�D�GI���  ������(��G=�  ���G����a�U�D�p����\�  �)\���G��
����P�
׍��Q ��I�Dq=zA\�fAq=�?q=^A\�
A��A���@�%@���q}�D)\�?{�Aq=�A��\@�(xA�z$A�+Aף�@ff�@��D33�@=
�@H�A=
�A�� AH�A{nA�uA��EAR �D
�wAR��A��AffB�zB�Q�AffB���A�z�AH�D��ף�@��5A
�OA=
�A33�A�mA=
�A33�A)��D�G���
¸��ףh�R�N�{.@R��?�p1��Qx?q��DHᚿ����)\��̢�  |�{b��G�?)\�>��D�="�D��1���D�H��;�������
����p��-���|D33S�ff��{���M�\�p�2�33 �R���Q��f�~D)\�@=
��\���q=����.�ףQ��®G���
�D��QA�̦A���@��\����\����(�ff������=��D��=A���A��B  �AffA{�@�����33?�),�D��@�G�Aq=�A=
B�z�A)\SA  @A\�^�=
��
�D��%@���@  �A�zBff B33�A��|A�piA�5���D��L��(�ff�@ff�A)\�A��B���A��IAq=6A{l�D�'@�z�ף @q=�@)\�A�Q�A{B\��A�sA
/�D\��@33A  �@�pAffNA  �A�zBff6B��B  �D�zhAH�A
׹Aq=�A���A�p�A�B��QB�pBᢇD�pQA���A��B�GB�zB
�B{"B�zQB���B���D������@�z�A��A{�A�z�A33�A
�Bq=<B�1�D=
���z8���?R��A)\�A�Q�AR��A�p�A���A3ۅD�p-�H��
�c�33��{VA��Aף�A=
�A�«A���DR���ff��R�.�
ׁ���H�q=6A�AR��A��A3;�Dף@�  ��R���H�^��������{A)\gAף�Af��D�̴�\�
�ff*���U�ף�������<�R�.@��Af��D  ��ff�\�N�ffn�H��ף�����ff��\�¿ͬ�D33#�����33G�)\w������G��=
������̔��&�D�z�A{�A�(PA��@33�@=
@33��q=��{>��p�D�z�A�zB�GB\��A)\�A�G�A)\�A��A
�;AR��D��@ff�A�p)Bq=Bq=B�G�A33�A�G�A���AHɊD�zhA�(�A�QB\�cB)\YB)\HB��1BR�%B��B�F�DH�z@���A��A  $Bq=sB=
iB=
XB�pABff5BH�D�µ�   @q=�A�(�A�QB\�mB)\cB)\RB��;B{ĊD��)�q=�����{fA���AR�B��bB��XB��GB)��DR��?��h�����?���AH�A�B��iBR�_B�T�D�(�A{�AH�A��AH�A\�B�B��eB ��BfV�D��L=\��A�z�A�G�A��A�G�A��BR�B��eB3c�DffA33A���A��A�z�A��A�z�A)\-B�Q>B
��D�QH�ף�@q=�@R��Aף�A�p�A{�A�p�A
� B ��D���@�̌@��LA��MA���A�pB��A�Q�A
�B��D�G�A��A�z�A
�B=
B�QB{XB�zMB��GBq�D�(@A�B�� B�GBH�5B{6B{��B\�B�}BHi�D�(���(�@�Q�A�GB�� B)\"B\�"BףmB��tB{ēD��-A=
�@
׏A\�B�8B�(,B��MB��MB��B�q�D�p�@�(�Aq=6A33�Aq=0B)\NB
�AB�pcBףcB\O�Dq=��H�@=
sA��$A\��A��+B=
JB�=B�_B�+�D���=
#�ף��\��@���?)\SA�zB��%B{B
ǓD33�@�Q��H��
ף=H�.A���@�z�AH�B  9BᲓD�G!�=
�@�z��=
�������$A�p�@�p�A)\Bf�D\�y�{|��h�\���R���{�)\P�H�c�
�3�R��D��I�������R8�¤p������\������Í�DR�zA{��Q��{���H��B�k��R����D�(����U��(�)\�¸�������f&��u��ff{Dq=G�����ף|�33��q�/ø�0��+�H�4���5��m�D�QB�?�  $�ף��=����(
�=�
ä�����уD{2A
�CB��Y�����33c��G��\���Q�¸���{ăD=
׾)\+A�(BB\����Q����i����f���{ �f�D�z�A��A�(�A33�B�Q�A�;��(|@  :®��k�D�G�AH�B33BR�?B��B��B���@�̴AR��� ��D�z�@ff�A�p%B��#B�GPB�̳B)\ B�1A���A��D�pA�CA��A��EB�DBףpB���BR�@B�G�A 8�D\�"A  �A��A33$B�pnB��lBף�B�L�B)\iBͼ�Dffv����@fffA�Q�A��B=
_B)\]B���B���B��D��A�peA  �A)\B��B\�]Bf�B\�B�Q�Bf��D��yA33B���A�p B��@B)\QB  �B��B�G�BR��D����G)���@ףp@R�^A{�A33�Aq=3B�z}Bq��D�pY���)\��)\���G����>R�A��HA���A�D=
�ff8�{���R�b����)\)�R� �R�������{��D�KAq=���z�q=n���/�{��H���������5�R؈D��)A�̺A\��  ����C��Q�q=r�����G�͜�D�z�@{�A���A�GA�H��33+����  ���M�Ha�Dq=bAq=�A33�A�.B��1A{��G����P��z�@),�D�peA
��A�z
B��4BH�gB��A��=A�e���?*
dtype0*
_output_shapes
:	�


z
MatMulMatMulMatMul/aVariable/read*
T0* 
_output_shapes
:
�
�*
transpose_a( *
transpose_b( 
N
addAddMatMulVariable_1/read* 
_output_shapes
:
�
�*
T0
<
ReluReluadd*
T0* 
_output_shapes
:
�
�
y
MatMul_1MatMulReluVariable_2/read*
T0*
_output_shapes
:	�
d*
transpose_a( *
transpose_b( 
Q
add_1AddMatMul_1Variable_3/read*
T0*
_output_shapes
:	�
d
?
Relu_1Reluadd_1*
T0*
_output_shapes
:	�
d
{
MatMul_2MatMulRelu_1Variable_4/read*
_output_shapes
:	�
2*
transpose_a( *
transpose_b( *
T0
Q
add_2AddMatMul_2Variable_5/read*
_output_shapes
:	�
2*
T0
?
Relu_2Reluadd_2*
T0*
_output_shapes
:	�
2
{
MatMul_3MatMulRelu_2Variable_6/read*
T0*
_output_shapes
:	�
*
transpose_a( *
transpose_b( 
Q
add_3AddMatMul_3Variable_7/read*
T0*
_output_shapes
:	�

R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
n
ArgMaxArgMaxadd_3ArgMax/dimension*

Tidx0*
T0*
output_type0	*
_output_shapes	
:�

J
softmax_tensorSoftmaxadd_3*
T0*
_output_shapes
:	�

̯
!softmax_cross_entropy_loss/Cast/xConst*�
value�B�	�
"Ю              �?      �?              �?                      �?      �?                      �?      �?                      �?              �?      �?                      �?      �?              �?                      �?              �?      �?              �?              �?                      �?              �?              �?              �?              �?              �?              �?              �?      �?                      �?              �?      �?                      �?      �?                      �?              �?      �?              �?              �?                      �?              �?      �?              �?                      �?              �?      �?              �?              �?                      �?      �?                      �?      �?                      �?              �?              �?      �?              �?              �?              �?                      �?              �?      �?              �?              �?              �?                      �?              �?      �?                      �?      �?                      �?              �?              �?              �?      �?              �?              �?                      �?      �?                      �?              �?              �?              �?      �?                      �?              �?      �?                      �?      �?              �?              �?              �?              �?                      �?              �?              �?              �?      �?              �?                      �?              �?      �?              �?                      �?      �?              �?              �?              �?                      �?              �?      �?              �?                      �?              �?              �?              �?      �?                      �?              �?              �?              �?      �?              �?                      �?              �?              �?              �?      �?                      �?              �?      �?              �?                      �?              �?              �?              �?              �?      �?                      �?      �?                      �?      �?              �?              �?              �?              �?                      �?              �?      �?              �?              �?                      �?      �?              �?                      �?              �?      �?              �?              �?              �?                      �?              �?              �?      �?              �?                      �?      �?              �?                      �?              �?              �?              �?      �?              �?                      �?              �?      �?                      �?              �?              �?      �?                      �?      �?                      �?      �?                      �?              �?      �?                      �?      �?              �?              �?              �?              �?                      �?      �?                      �?              �?      �?              �?                      �?              �?              �?              �?      �?              �?                      �?      �?                      �?              �?              �?      �?              �?              �?              �?                      �?              �?              �?              �?      �?                      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?                      �?      �?              �?              �?              �?              �?                      �?      �?                      �?              �?      �?              �?              �?                      �?      �?              �?              �?              �?              �?              �?                      �?      �?                      �?              �?      �?              �?              �?              �?                      �?      �?                      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?                      �?      �?                      �?      �?                      �?      �?                      �?      �?              �?                      �?              �?      �?              �?              �?              �?                      �?              �?              �?              �?              �?              �?      �?                      �?      �?                      �?              �?      �?              �?              �?                      �?      �?                      �?      �?              �?              �?                      �?      �?                      �?      �?                      �?              �?              �?      �?              �?              �?              �?                      �?      �?                      �?              �?              �?      �?              �?              �?                      �?      �?                      �?              �?              �?      �?              �?              �?                      �?      �?                      �?              �?              �?              �?      �?                      �?              �?      �?                      �?      �?              �?                      �?              �?              �?      �?              �?              �?                      �?      �?                      �?      �?                      �?              �?      �?              �?              �?                      �?      �?                      �?              �?      �?                      �?              �?              �?              �?      �?                      �?      �?                      �?              �?              �?      �?                      �?      �?                      �?              �?              �?      �?              �?              �?                      �?              �?              �?      �?              �?              �?                      �?      �?              �?                      �?              �?      �?              �?              �?              �?                      �?      �?              �?                      �?      �?                      �?              �?      �?              �?              �?                      �?              �?      �?              �?                      �?              �?              �?      �?              �?                      �?      �?              �?              �?              �?                      �?      �?                      �?      �?                      �?      �?              �?                      �?              �?              �?              �?      �?              �?              �?                      �?              �?              �?              �?              �?              �?      �?                      �?              �?      �?              �?              �?                      �?              �?              �?      �?              �?              �?                      �?      �?              �?                      �?              �?              �?              �?      �?                      �?      �?                      �?      �?                      �?      �?                      �?      �?              �?                      �?      �?                      �?              �?      �?              �?              �?                      �?      �?                      �?              �?      �?              �?                      �?      �?                      �?      �?              �?                      �?      �?                      �?      �?                      �?      �?              �?                      �?              �?      �?                      �?              �?      �?              �?                      �?      �?              �?              �?              �?                      �?              �?              �?              �?              �?              �?      �?                      �?      �?              �?              �?                      �?              �?              �?              �?      �?              �?                      �?      �?                      �?      �?                      �?      �?                      �?              �?              �?      �?                      �?      �?              �?              �?                      �?      �?              �?              �?                      �?              �?              �?              �?      �?                      �?              �?      �?                      �?              �?      �?                      �?      �?              �?                      �?              �?              �?      �?              �?              �?              �?              �?                      �?      �?                      �?              �?              �?      �?                      �?      �?              �?              �?                      �?      �?                      �?              �?      �?                      �?      �?              �?              �?                      �?              �?              �?              �?              �?      �?                      �?      �?              �?              �?              �?              �?              �?              �?                      �?      �?              �?                      �?              �?      �?                      �?      �?                      �?              �?              �?      �?              �?              �?                      �?              �?      �?                      �?      �?                      �?              �?      �?                      �?      �?                      �?              �?      �?              �?                      �?      �?              �?                      �?              �?              �?              �?              �?              �?      �?                      �?      �?                      �?      �?              �?                      �?      �?                      �?      �?                      �?      �?                      �?              �?              �?      �?              �?              �?                      �?      �?              �?              �?                      �?              �?              �?      �?                      �?              �?      �?                      �?              �?      �?                      �?      �?                      �?              �?              �?      �?              �?              �?              �?                      �?              �?      �?              �?                      �?              �?      �?              �?                      �?              �?      �?                      �?              �?              �?              �?      �?              �?                      �?      �?              �?                      �?              �?              �?      �?              �?              �?              �?                      �?      �?              �?              �?                      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?                      �?              �?      �?                      �?      �?                      �?              �?              �?              �?      �?                      �?              �?      �?                      �?      �?              �?              �?                      �?              �?              �?              �?      �?                      �?      �?                      �?      �?              �?              �?                      �?              �?      �?                      �?              �?      �?                      �?      �?              �?              �?              �?                      �?              �?      �?              �?              �?                      �?      �?              �?                      �?      �?                      �?              �?      �?                      �?              �?              �?      �?                      �?      �?                      �?              �?      �?              �?              �?                      �?              �?              �?              �?      �?              �?              �?                      �?              �?              �?      �?                      �?              �?              �?      �?              �?                      �?              �?              �?              �?              �?              �?      �?                      �?              �?              �?              �?              �?      �?                      �?              �?      �?              �?                      �?      �?                      �?      �?                      �?      �?              �?              �?                      �?      �?                      �?      �?              �?                      �?      �?                      �?      �?              �?              �?                      �?      �?                      �?      �?              �?                      �?              �?              �?      �?                      �?              �?      �?              �?              �?                      �?      �?              �?              �?                      �?              �?      �?                      �?      �?                      �?      �?                      �?              �?      �?                      �?      �?                      �?              �?              �?              �?      �?              �?              �?                      �?      �?                      �?      �?                      �?              �?              �?      �?                      �?              �?      �?                      �?              �?      �?              �?                      �?      �?              �?              �?              �?                      �?              �?              �?              �?              �?              �?              �?              �?      �?                      �?      �?              �?              �?              �?                      �?              �?      �?                      �?      �?              �?                      �?      �?              �?              �?                      �?              �?              �?              �?              �?      �?              �?                      �?      �?                      �?      �?              �?              �?                      �?              �?      �?                      �?              �?              �?              �?              �?      �?              �?              �?                      �?      �?                      �?      �?                      �?      �?                      �?              �?      �?                      �?              �?      �?              �?              �?              �?                      �?              �?      �?              �?              �?                      �?      �?              �?              �?                      �?              �?              �?              �?      �?              �?                      �?              �?              �?              �?      �?                      �?      �?                      �?              �?              �?      �?                      �?      �?                      �?              �?              �?      �?                      �?      �?                      �?              �?              �?              �?              �?      �?              �?              �?                      �?      �?                      �?              �?      �?              �?              �?              �?                      �?      �?              �?              �?              �?              �?              �?                      �?              �?      �?              �?                      �?              �?              �?      �?              �?              �?                      �?      �?                      �?              �?      �?              �?                      �?              �?              �?      �?              �?                      �?      �?              �?                      �?      �?                      �?      �?                      �?      �?                      �?      �?                      �?      �?                      �?              �?              �?              �?      �?                      �?              �?      �?              �?              �?                      �?      �?              �?              �?              �?              �?                      �?              �?      �?              �?              �?              �?                      �?              �?              �?      �?                      �?              �?      �?                      �?              �?              �?      �?              �?                      �?              �?      �?                      �?              �?              �?      �?                      �?              �?      �?              �?                      �?              �?      �?              �?              �?                      �?      �?              �?              �?                      �?      �?                      �?              �?              �?      �?                      �?      �?                      �?      �?                      �?      �?                      �?              �?              �?              �?      �?              �?              �?              �?              �?                      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?                      �?              �?              �?              �?      �?              �?                      �?      �?                      �?      �?              �?              �?                      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?                      �?              �?              �?      �?              �?                      �?      �?              �?              �?              �?              �?              �?                      �?      �?                      �?      �?                      �?              �?      �?                      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?                      �?      �?              �?                      �?      �?                      �?              �?      �?                      �?              �?              �?              �?              �?              �?              �?              �?      �?                      �?              �?              �?      �?                      �?              �?      �?              �?                      �?      �?              �?              �?                      �?      �?                      �?      �?                      �?      �?              �?                      �?      �?              �?              �?                      �?      �?                      �?              �?              �?              �?              �?              �?      �?                      �?              �?      �?                      �?              �?      �?              �?              �?                      �?      �?                      �?              �?      �?                      �?      �?              �?              �?              �?                      �?              �?      �?                      �?      �?              �?              �?                      �?              �?      �?              �?              �?                      �?              �?              �?      �?              �?                      �?              �?      �?                      �?              �?              �?      �?              �?              �?                      �?              �?              �?      �?              �?                      �?              �?              �?              �?      �?                      �?      �?                      �?              �?      �?              �?                      �?      �?                      �?              �?              �?              �?      �?                      �?      �?                      �?              �?      �?                      �?      �?              �?                      �?              �?              �?      �?                      �?      �?              �?              �?                      �?      �?                      �?      �?              �?                      �?              �?              �?              �?      �?              �?              �?              �?              �?                      �?              �?              �?              �?              �?      �?                      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?                      �?              �?              �?              �?              �?      �?              �?                      �?              �?              �?              �?      �?                      �?              �?              �?      �?                      �?              �?      �?              �?                      �?      �?              �?              �?                      �?      �?              �?                      �?              �?      �?                      �?              �?              �?              �?              �?      �?                      �?              �?      �?              �?              �?                      �?              �?              �?              �?              �?              �?*
dtype0*
_output_shapes
:	�

�
softmax_cross_entropy_loss/CastCast!softmax_cross_entropy_loss/Cast/x*

SrcT0*
_output_shapes
:	�
*

DstT0
�
8softmax_cross_entropy_loss/xentropy/labels_stop_gradientStopGradientsoftmax_cross_entropy_loss/Cast*
T0*
_output_shapes
:	�

j
(softmax_cross_entropy_loss/xentropy/RankConst*
dtype0*
_output_shapes
: *
value	B :
z
)softmax_cross_entropy_loss/xentropy/ShapeConst*
valueB"u     *
dtype0*
_output_shapes
:
l
*softmax_cross_entropy_loss/xentropy/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
|
+softmax_cross_entropy_loss/xentropy/Shape_1Const*
valueB"u     *
dtype0*
_output_shapes
:
k
)softmax_cross_entropy_loss/xentropy/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'softmax_cross_entropy_loss/xentropy/SubSub*softmax_cross_entropy_loss/xentropy/Rank_1)softmax_cross_entropy_loss/xentropy/Sub/y*
T0*
_output_shapes
: 
�
/softmax_cross_entropy_loss/xentropy/Slice/beginPack'softmax_cross_entropy_loss/xentropy/Sub*
N*
_output_shapes
:*
T0*

axis 
x
.softmax_cross_entropy_loss/xentropy/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
)softmax_cross_entropy_loss/xentropy/SliceSlice+softmax_cross_entropy_loss/xentropy/Shape_1/softmax_cross_entropy_loss/xentropy/Slice/begin.softmax_cross_entropy_loss/xentropy/Slice/size*
T0*
Index0*
_output_shapes
:
�
3softmax_cross_entropy_loss/xentropy/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
q
/softmax_cross_entropy_loss/xentropy/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
*softmax_cross_entropy_loss/xentropy/concatConcatV23softmax_cross_entropy_loss/xentropy/concat/values_0)softmax_cross_entropy_loss/xentropy/Slice/softmax_cross_entropy_loss/xentropy/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
+softmax_cross_entropy_loss/xentropy/ReshapeReshapeadd_3*softmax_cross_entropy_loss/xentropy/concat*
T0*
Tshape0*
_output_shapes
:	�

l
*softmax_cross_entropy_loss/xentropy/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
|
+softmax_cross_entropy_loss/xentropy/Shape_2Const*
valueB"u     *
dtype0*
_output_shapes
:
m
+softmax_cross_entropy_loss/xentropy/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
)softmax_cross_entropy_loss/xentropy/Sub_1Sub*softmax_cross_entropy_loss/xentropy/Rank_2+softmax_cross_entropy_loss/xentropy/Sub_1/y*
T0*
_output_shapes
: 
�
1softmax_cross_entropy_loss/xentropy/Slice_1/beginPack)softmax_cross_entropy_loss/xentropy/Sub_1*
T0*

axis *
N*
_output_shapes
:
z
0softmax_cross_entropy_loss/xentropy/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
+softmax_cross_entropy_loss/xentropy/Slice_1Slice+softmax_cross_entropy_loss/xentropy/Shape_21softmax_cross_entropy_loss/xentropy/Slice_1/begin0softmax_cross_entropy_loss/xentropy/Slice_1/size*
T0*
Index0*
_output_shapes
:
�
5softmax_cross_entropy_loss/xentropy/concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
s
1softmax_cross_entropy_loss/xentropy/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
,softmax_cross_entropy_loss/xentropy/concat_1ConcatV25softmax_cross_entropy_loss/xentropy/concat_1/values_0+softmax_cross_entropy_loss/xentropy/Slice_11softmax_cross_entropy_loss/xentropy/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
-softmax_cross_entropy_loss/xentropy/Reshape_1Reshape8softmax_cross_entropy_loss/xentropy/labels_stop_gradient,softmax_cross_entropy_loss/xentropy/concat_1*
T0*
Tshape0*
_output_shapes
:	�

�
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits+softmax_cross_entropy_loss/xentropy/Reshape-softmax_cross_entropy_loss/xentropy/Reshape_1*
T0*&
_output_shapes
:�
:	�

m
+softmax_cross_entropy_loss/xentropy/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
)softmax_cross_entropy_loss/xentropy/Sub_2Sub(softmax_cross_entropy_loss/xentropy/Rank+softmax_cross_entropy_loss/xentropy/Sub_2/y*
_output_shapes
: *
T0
{
1softmax_cross_entropy_loss/xentropy/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
0softmax_cross_entropy_loss/xentropy/Slice_2/sizePack)softmax_cross_entropy_loss/xentropy/Sub_2*
N*
_output_shapes
:*
T0*

axis 
�
+softmax_cross_entropy_loss/xentropy/Slice_2Slice)softmax_cross_entropy_loss/xentropy/Shape1softmax_cross_entropy_loss/xentropy/Slice_2/begin0softmax_cross_entropy_loss/xentropy/Slice_2/size*
T0*
Index0*#
_output_shapes
:���������
�
-softmax_cross_entropy_loss/xentropy/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy+softmax_cross_entropy_loss/xentropy/Slice_2*
_output_shapes	
:�
*
T0*
Tshape0
|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B : 
�
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeConst*
valueB:�
*
dtype0*
_output_shapes
:
}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
�
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
softmax_cross_entropy_loss/MulMul-softmax_cross_entropy_loss/xentropy/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*
_output_shapes	
:�
*
T0
�
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
:*
valueB: 
�
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
T0*
_output_shapes
: 
�
Asoftmax_cross_entropy_loss/num_present/zeros_like/shape_as_tensorConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
7softmax_cross_entropy_loss/num_present/zeros_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1softmax_cross_entropy_loss/num_present/zeros_likeFillAsoftmax_cross_entropy_loss/num_present/zeros_like/shape_as_tensor7softmax_cross_entropy_loss/num_present/zeros_like/Const*
T0*

index_type0*
_output_shapes
: 
�
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
�
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
T0*
_output_shapes
: 
�
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB:�
*
dtype0*
_output_shapes
:
�
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B :
�
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB:�
*
dtype0*
_output_shapes
:
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*
_output_shapes	
:�

�
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*
_output_shapes	
:�

�
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
:*
valueB: 
�
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
_output_shapes
: *
T0
�
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
T0*
_output_shapes
: 
�
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
�
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
_output_shapes
: *
T0
�
softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
_output_shapes
: *
T0
�
5softmax_cross_entropy_loss/zeros_like/shape_as_tensorConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
+softmax_cross_entropy_loss/zeros_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
%softmax_cross_entropy_loss/zeros_likeFill5softmax_cross_entropy_loss/zeros_like/shape_as_tensor+softmax_cross_entropy_loss/zeros_like/Const*
T0*

index_type0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
T0*
_output_shapes
: 
^
OptimizeLoss/tagsConst*
dtype0*
_output_shapes
: *
valueB BOptimizeLoss
s
OptimizeLossScalarSummaryOptimizeLoss/tags softmax_cross_entropy_loss/value*
T0*
_output_shapes
: 
�
,mean/total/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
�
"mean/total/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@mean/total
�
mean/total/Initializer/zerosFill,mean/total/Initializer/zeros/shape_as_tensor"mean/total/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@mean/total*
_output_shapes
: 
�

mean/total
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@mean/total*
	container *
shape: 
�
mean/total/AssignAssign
mean/totalmean/total/Initializer/zeros*
T0*
_class
loc:@mean/total*
validate_shape(*
_output_shapes
: *
use_locking(
g
mean/total/readIdentity
mean/total*
T0*
_class
loc:@mean/total*
_output_shapes
: 
�
,mean/count/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
"mean/count/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
mean/count/Initializer/zerosFill,mean/count/Initializer/zeros/shape_as_tensor"mean/count/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@mean/count*
_output_shapes
: 
�

mean/count
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@mean/count*
	container *
shape: 
�
mean/count/AssignAssign
mean/countmean/count/Initializer/zeros*
use_locking(*
T0*
_class
loc:@mean/count*
validate_shape(*
_output_shapes
: 
g
mean/count/readIdentity
mean/count*
T0*
_class
loc:@mean/count*
_output_shapes
: 
K
	mean/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Q
mean/ToFloat_1Cast	mean/Size*

SrcT0*
_output_shapes
: *

DstT0
M

mean/ConstConst*
valueB *
dtype0*
_output_shapes
: 
{
mean/SumSum softmax_cross_entropy_loss/value
mean/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
mean/AssignAdd	AssignAdd
mean/totalmean/Sum*
use_locking( *
T0*
_class
loc:@mean/total*
_output_shapes
: 
�
mean/AssignAdd_1	AssignAdd
mean/countmean/ToFloat_1!^softmax_cross_entropy_loss/value*
use_locking( *
T0*
_class
loc:@mean/count*
_output_shapes
: 
Z
mean/truedivRealDivmean/total/readmean/count/read*
T0*
_output_shapes
: 
b
mean/zeros_like/shape_as_tensorConst*
dtype0*
_output_shapes
: *
valueB 
Z
mean/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
mean/zeros_likeFillmean/zeros_like/shape_as_tensormean/zeros_like/Const*
T0*

index_type0*
_output_shapes
: 
Z
mean/GreaterGreatermean/count/readmean/zeros_like*
T0*
_output_shapes
: 
b

mean/valueSelectmean/Greatermean/truedivmean/zeros_like*
T0*
_output_shapes
: 
\
mean/truediv_1RealDivmean/AssignAddmean/AssignAdd_1*
T0*
_output_shapes
: 
d
!mean/zeros_like_1/shape_as_tensorConst*
dtype0*
_output_shapes
: *
valueB 
\
mean/zeros_like_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
mean/zeros_like_1Fill!mean/zeros_like_1/shape_as_tensormean/zeros_like_1/Const*
T0*

index_type0*
_output_shapes
: 
_
mean/Greater_1Greatermean/AssignAdd_1mean/zeros_like_1*
T0*
_output_shapes
: 
l
mean/update_opSelectmean/Greater_1mean/truediv_1mean/zeros_like_1*
T0*
_output_shapes
: 
#

group_depsNoOp^mean/update_op
�
+eval_step/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@eval_step*
dtype0*
_output_shapes
: 
�
!eval_step/Initializer/zeros/ConstConst*
value	B	 R *
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
�
eval_step/Initializer/zerosFill+eval_step/Initializer/zeros/shape_as_tensor!eval_step/Initializer/zeros/Const*
T0	*

index_type0*
_class
loc:@eval_step*
_output_shapes
: 
�
	eval_step
VariableV2*
_class
loc:@eval_step*
	container *
shape: *
dtype0	*
_output_shapes
: *
shared_name 
�
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@eval_step
d
eval_step/readIdentity	eval_step*
_output_shapes
: *
T0	*
_class
loc:@eval_step
Q
AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@eval_step
U
readIdentity	eval_step^group_deps
^AssignAdd*
T0	*
_output_shapes
: 
;
IdentityIdentityread*
_output_shapes
: *
T0	
�
initNoOp^global_step/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedVariable*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitialized
Variable_1*
dtype0*
_output_shapes
: *
_class
loc:@Variable_1
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitialized
Variable_2*
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitialized
Variable_3*
dtype0*
_output_shapes
: *
_class
loc:@Variable_3
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitialized
Variable_4*
_class
loc:@Variable_4*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized
Variable_5*
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized
Variable_6*
dtype0*
_output_shapes
: *
_class
loc:@Variable_6
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized
Variable_7*
_class
loc:@Variable_7*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitialized
mean/total*
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitialized
mean/count*
dtype0*
_output_shapes
: *
_class
loc:@mean/count
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitialized	eval_step*
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
�
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_11"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:
�
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
�
$report_uninitialized_variables/ConstConst"/device:CPU:0*�
value�B�Bglobal_stepBVariableB
Variable_1B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7B
mean/totalB
mean/countB	eval_step*
dtype0*
_output_shapes
:
�
1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask 
�
3report_uninitialized_variables/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_2StridedSlice3report_uninitialized_variables/boolean_mask/Shape_2Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackCreport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: 
�
;report_uninitialized_variables/boolean_mask/concat/values_1Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/strided_slice_1;report_uninitialized_variables/boolean_mask/concat/values_1;report_uninitialized_variables/boolean_mask/strided_slice_27report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*

Tidx0*
T0*
N*
_output_shapes
:
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
_output_shapes
:*
T0
*
Tshape0
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:���������*
T0

�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze"/device:CPU:0*#
_output_shapes
:���������*
Tindices0	*
Tparams0*
validate_indices(
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*

Tidx0*
T0*
N*#
_output_shapes
:���������
�
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_output_shapes
: *
_class
loc:@global_step
�
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedVariable*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitialized
Variable_1*
dtype0*
_output_shapes
: *
_class
loc:@Variable_1
�
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitialized
Variable_2*
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitialized
Variable_3*
_class
loc:@Variable_3*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitialized
Variable_4*
_class
loc:@Variable_4*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized
Variable_5*
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitialized
Variable_6*
_class
loc:@Variable_6*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitialized
Variable_7*
dtype0*
_output_shapes
: *
_class
loc:@Variable_7
�
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_8"/device:CPU:0*
N	*
_output_shapes
:	*
T0
*

axis 
�
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:	
�
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*~
valueuBs	Bglobal_stepBVariableB
Variable_1B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7*
dtype0*
_output_shapes
:	
�
3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:	
�
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
�
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:	*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
�
5report_uninitialized_variables_1/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:	*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_2StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_2Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
T0*
Index0
�
=report_uninitialized_variables_1/boolean_mask/concat/values_1Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
N*
_output_shapes
:*
T0*

axis 
�
9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/strided_slice_1=report_uninitialized_variables_1/boolean_mask/concat/values_1=report_uninitialized_variables_1/boolean_mask/strided_slice_29report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*
T0*
N*
_output_shapes
:*

Tidx0
�
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
_output_shapes
:	*
T0*
Tshape0
�
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
�
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:	
�
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:���������
�
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*
T0	*#
_output_shapes
:���������*
squeeze_dims

�
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze"/device:CPU:0*#
_output_shapes
:���������*
Tindices0	*
Tparams0*
validate_indices(
I
init_2NoOp^mean/total/Assign^mean/count/Assign^eval_step/Assign

init_all_tablesNoOp

init_3NoOp
8
group_deps_2NoOp^init_2^init_all_tables^init_3
Q
Merge/MergeSummaryMergeSummaryOptimizeLoss*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_27990c85253b44348e02170c5b3644c2/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:	*~
valueuBs	BVariableB
Variable_1B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7Bglobal_step
�
save/SaveV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable
Variable_1
Variable_2
Variable_3
Variable_4
Variable_5
Variable_6
Variable_7global_step"/device:CPU:0*
dtypes
2		
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
�
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*~
valueuBs	BVariableB
Variable_1B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7Bglobal_step*
dtype0*
_output_shapes
:	
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2		
�
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	
�
�
save/Assign_1Assign
Variable_1save/RestoreV2:1*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@Variable_1
�
save/Assign_2Assign
Variable_2save/RestoreV2:2*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*
_class
loc:@Variable_2
�
save/Assign_3Assign
Variable_3save/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:d
�
save/Assign_4Assign
Variable_4save/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:d2
�
save/Assign_5Assign
Variable_5save/RestoreV2:5*
validate_shape(*
_output_shapes
:2*
use_locking(*
T0*
_class
loc:@Variable_5
�
save/Assign_6Assign
Variable_6save/RestoreV2:6*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:2
�
save/Assign_7Assign
Variable_7save/RestoreV2:7*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_8Assignglobal_stepsave/RestoreV2:8*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8
-
save/restore_allNoOp^save/restore_shard"��ږL     �PX�	K�8�j��AJ��
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
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
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
�
Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
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
N
IsVariableInitialized
ref"dtype�
is_initialized
"
dtypetype�


LogicalNot
x

y

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
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
shared_namestring �
E
Where

input"T	
index	"%
Ttype0
:
2	
*1.6.02v1.6.0-0-gd2e24b6039��
�
-global_step/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@global_step*
dtype0*
_output_shapes
: 
�
#global_step/Initializer/zeros/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R *
_class
loc:@global_step
�
global_step/Initializer/zerosFill-global_step/Initializer/zeros/shape_as_tensor#global_step/Initializer/zeros/Const*
T0	*

index_type0*
_class
loc:@global_step*
_output_shapes
: 
�
global_step
VariableV2*
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step
g
truncated_normal/shapeConst*
valueB"
   �   *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
truncated_normal/stddevConst*
valueB
 *ↁ=*
dtype0*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
T0*
dtype0*
_output_shapes
:	
�*
seed2 *

seed 
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*
_output_shapes
:	
�
n
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes
:	
�
~
Variable
VariableV2*
shared_name *
dtype0*
_output_shapes
:	
�*
	container *
shape:	
�
�
Variable/AssignAssignVariabletruncated_normal*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	
�
j
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
:	
�
`
zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:�
P
zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
i
zerosFillzeros/shape_as_tensorzeros/Const*
T0*

index_type0*
_output_shapes	
:�
x

Variable_1
VariableV2*
shared_name *
dtype0*
_output_shapes	
:�*
	container *
shape:�
�
Variable_1/AssignAssign
Variable_1zeros*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes	
:�*
use_locking(
l
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes	
:�
i
truncated_normal_1/shapeConst*
valueB"�   d   *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���=
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
T0*
dtype0*
_output_shapes
:	�d*
seed2 *

seed 
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
_output_shapes
:	�d*
T0
t
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*
_output_shapes
:	�d
�

Variable_2
VariableV2*
shape:	�d*
shared_name *
dtype0*
_output_shapes
:	�d*
	container 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*
_class
loc:@Variable_2
p
Variable_2/readIdentity
Variable_2*
_output_shapes
:	�d*
T0*
_class
loc:@Variable_2
a
zeros_1/shape_as_tensorConst*
valueB:d*
dtype0*
_output_shapes
:
R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n
zeros_1Fillzeros_1/shape_as_tensorzeros_1/Const*
_output_shapes
:d*
T0*

index_type0
v

Variable_3
VariableV2*
dtype0*
_output_shapes
:d*
	container *
shape:d*
shared_name 
�
Variable_3/AssignAssign
Variable_3zeros_1*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:d
k
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:d
i
truncated_normal_2/shapeConst*
valueB"d   2   *
dtype0*
_output_shapes
:
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *��>
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
T0*
dtype0*
_output_shapes

:d2*
seed2 *

seed 
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
dtype0*
_output_shapes

:d2*
	container *
shape
:d2*
shared_name 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:d2
o
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*
_output_shapes

:d2
a
zeros_2/shape_as_tensorConst*
valueB:2*
dtype0*
_output_shapes
:
R
zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n
zeros_2Fillzeros_2/shape_as_tensorzeros_2/Const*
T0*

index_type0*
_output_shapes
:2
v

Variable_5
VariableV2*
dtype0*
_output_shapes
:2*
	container *
shape:2*
shared_name 
�
Variable_5/AssignAssign
Variable_5zeros_2*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:2
k
Variable_5/readIdentity
Variable_5*
_output_shapes
:2*
T0*
_class
loc:@Variable_5
i
truncated_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"2      
\
truncated_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_3/stddevConst*
valueB
 *�5?*
dtype0*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
T0*
dtype0*
_output_shapes

:2*
seed2 *

seed 
�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
_output_shapes

:2*
T0
s
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes

:2
~

Variable_6
VariableV2*
shape
:2*
shared_name *
dtype0*
_output_shapes

:2*
	container 
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:2
o
Variable_6/readIdentity
Variable_6*
_output_shapes

:2*
T0*
_class
loc:@Variable_6
a
zeros_3/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
R
zeros_3/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
n
zeros_3Fillzeros_3/shape_as_tensorzeros_3/Const*
T0*

index_type0*
_output_shapes
:
v

Variable_7
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
Variable_7/AssignAssign
Variable_7zeros_3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@Variable_7
k
Variable_7/readIdentity
Variable_7*
_output_shapes
:*
T0*
_class
loc:@Variable_7
��
MatMul/aConst*�
valueݴBٴ	�

"ȴ�7(D\��@R��@��MA���A�(�A��A�QBff�A�Q�A�H)Dף�@��A�A=
�A\�B�(BףBffB�G	B��(D��̿H�*@   A�(�@�zxA�(B��Bq=B  B�`'D�����(��=
W��(�?��Y?  A{�A�G�Aq=�A�K)D�p�@���?
�#=��@q=A�QA)\�AR�B�QB33)D\�¾�G�@�G�?{���{@�(Aq=A�Q�A33BR�)Dq=�?���?��A��E@R��?�Q�@�p1A�A���AN'D\��\���R����z��q=���p����i�=
�?)\? P)D�z Aף��ff�>)\�=��@H��?�G�=�(�@)\A� ,D�(,A�Q�A{A)\3A�G-A  �A�GA��-Aq=rA�k*D\����@)\GA33C@�(�@  �@R�BA�z�@�G�@�E+Dq=Z@H�:�H��@��}AR��@ףA��@�GyA��A�B*D����
�#�=
��\�r@�=Aq=@��@=
w@�z8A.*D
ף�
׋���u��G��{^@  8A��@H�z@\�b@��.D�p�AH�A��dA�Aq=6A33�A�p�A�(�A�̴A��0D  �@�p�AH��A�z�A���A��A33�AR�B�(�AH1/D�(��)\�?ff�A
םAH�zAR��A�(LA�(�Aff�A)-D�G	�)\_�R����7Aff2A33�@�((A�@=
oAR�,D���q=��Qt��Q�\�"A�pA�G�@33A�7@��0D\�rA��]Aף�@�G��z�@\��A  �A���AH��A�k1D��h@ff�A��A\�A��a@�z$A��A��AR��A�~1D���>�(|@�̘A�Q�A)\A�u@�G)A{�A��A�3D��A��	A
�CA33�AR��A{�A{BA=
�Aq=B �5D�A�(�A\��A��A�zBq=B
��A�̪A���Av6D�?�bA��A(~�A���AG�B״B���A���A\7D��@��@)\{A{�A�z�A���A�p,B33'BH�BRX;Dq=vAq�A=
�A���A��B��B)\-B  jB��dB=J;D�Ga�R�rA�X�A�G�A=
�AR�B��B�z,B�iBq]<D���@\��@�AH��A��AR�B��-B�/B�=B  =D\�B@H��@
��@{�A4�A  �AH�B{:B�G;B �<D   ���?H�@
ף@{�A4�A  �AH�B{2B�q>D���@���@�A��IAffFA�Q�AR�B�	B  .B\?=D�G��)\@H��>��a@\��@��@  �A���A���A �>D�Q�@�Q8?   A  �@ףA�pUA��QA{�A��	B3@D���@��0Aף�@��XA��8A�piA��A)\�Aq=B��?D��̾�̤@\�*A
׻@ffRAff2A=
cA��A�(�A�u=D���)\#������Y?�(|���U@��?�(�@H�
A�:D  \��z������G��ffN����\�&�\�F�����#:Dף�>�zT�R���������H�F���=
�=
?���;D�p�@�z�@���)\��\����Q0��Q����$��GA� 0:D)\��\�B>��(?�pQ�33��ff��  ��
�C�q=���>9D�Gq�  (��e�=
G�H��)\��\����(��{���,:D{n@��L�����)\>�(?q=R������̺�ff��\�<Dף,A�(hA
�+A�Q�@H�.Aff6Aff�\�B���H� �-D��q���F�7�  G�a�q=F�)\D�)\{��L�r*D)\S�ff��ף{�l�
�{�Ha��{{�33y���)D�G�ף��ף��\��q=y��(�¸��®ǃ�
ׂ�f*D=
�?�Q��ffj��G��3����r��̀B�k��WS)D(C���������'`�¬ˆ��~�F��	[�¤p)D�d�>��%���h���������u��H��H�|�����(D=
'���	�ff���Ga��z���̞���������*D$��@
h"@��?@b�V�2U�?���=k��b��΀���+DW[�@=
GA�GAԚ$A��@R�A���@����̇���+D��>Q��@�ILA+�"AQ�)A-�@��A��@����q�*D���(���d*@���@ff�@��@=
'@�G�@�?n*DR���V��
׻�4�?q=�@�p}@�^�@)\�?33C@��&D��i��p}�Nb���£�s�R��z �q=*�K�"��S��#D�Gm��p��)\����33��H�����������%D
�+AH������p������z��$����A��k��y&D�7@��YA�(���}��z���$������\f�  �h�$D����~�"l�@���U�� A�����^K������3##D7����U��'���>�i�)\���G���x��(¤�!D�G���<�����Q|�����H��
�	����ˡ"�!D
ף�
׫���=�����}��������(
¸�q'Dq=�A���A\�zAA�A
�@�¥@R�~A��?�(X�~'D�G�?�Q�A��A)\�A6�.Aq=�@{�@�p�Aff6@�w&D33�������AH�A�UA:#�@������5@�GYA�&Dff@   ���u��Q�A��AR�vA6�A�z@{�@�I%Dq=��=
���������_Aq=^A��	A�1@  ��q�'D��A)\/@H�@�p=?   @q=�A���A�G�AA�:A��*D)\OA�(�A33{Aff�A33[A)\wA��BףB���A��,D�A�p�A���A)\�A�(�A)\�A�p�A
�2B�2B��.D)\�@33kA�G�AH�B33�A  B33�A�G�A��LB �-D�G�=
W@�G9A�Q�AffBq=�A=
�Aq=�A�Q�A��,D=
��H����(\���@\��A=
�A�z�A�G�A�z�A{�+D�pM�������(�q=���̄@H�A)\�A�̞A���AR�,D
�S@���=
׃����\�B�R��@)\�A
��A�G�Aq+D�p��=
g�q=��ף0�\�b�����q=
?  XA�z�A�Z+DH�?R����������G�33O�=
��  �?)\kAR8.D)\7AR�JA  �@��A33�@ף�?R�����@)\SA�c.D{.?q=BA��UA���@
�A���@
�@)\O��p�@ͬ/D�z�@q=�@q=�A��A�9A{nAR�:Aff�@\��@�}/D�<��@���@yX�A��A�T-A�JbAc�.A���@�14DΈ�Aף�A�¹A33�A�pB�GB33�A
�B  �A{D4D�z�>�ژA���A{�A��A��B�pB��A  B
4D��u��+��,�A�G�Aff�A
׹A��
B��B
��A
�4D  @��?q=�?�,�A�G�Aff�A
��A��B��BR�2D)\��)\��{���̤�6�ZAH�NA\��A  �A��A `1D�(����M���)��9��z4�w-�@���@=
?A��IA�71D�G!��Q��
�W�
�3�33C�\�>���@�p�@��4A��0D�%��pM��p�\����]��zl�
�g�%u�@��e@� /D�������
����y��̲��̠��z���(��u��R�0D
��@��?R�����  ��o��K�=
[�ffV� �4D��}A��A��A{fA  \A
��@
�c?��H@�@��4D=
�>�Q�A�G�A=
�A��lAR�bAף A��?
�c@~8D��dA�kAR��A
�B�p�A���A���AR��A��yA �7D{N��G1A  8A���A��
B��A=
�A  �A���A3S7D�����p��{A�� A)\�A�(B{�A�p�Aff�A��8D-�@P�W@P�>+gAT�mAm��A�nB���A���A�^9D+W@H�A)\�@ף`@�z�A
בAffBH�%B��B\�8DR������?{�@��@�G@\��A��A�pB��B �4D���
ב�T�m��� �  8��k�=
׾    ��}A�;5D=
�?��p�ff��S������ףP��G�?=
�?)�2D
���� �ff���Q���l��H���z��q=����{�1D�w���Y�H�>�)\���G��Nb��
ׯ��p��33��� 0D����H�*�)\��������H�n��Q��������/D��ѿ33��E��z��=
���z¤p���¤p�� `9D�B��B�p�A�z�A\��A  �A�G�?
ף<�rX@u<D�GEA
�KB�GEB=
*B\�B33�Aף�A�p]A��EA�j<D
�#�R�BA33KBףDBff)B��B���A)\�AH�ZA��;D�(<�ffF��A�p?BH�8BףB�(Bff�A
��A�k<D��<@
�#<���H�BAq=KB�DB�p)B��B  �Au<D��>ffF@
�#>    �GEA
�KB�GEB=
*B\�B)�<D{�?ף @R��@�G@{�?=
cA�GSBR�LB�z1Bf�AD�G�A�(�A)\�A���A�p�A�(�AffB�u�B.�BH�=Dף���U@{�@H�@ףA33�@{�@�(�A��`B\o?D=
�@����� A\�>A��@A�(pA�AA\�>A���A�@D�@ff2A�­��gAR��A��A��A  �AR��A�|AD��q@)\AH�nA33ӿ{�A���A�(�A�¹Aq=�A�WDDR�6A33sA=
�A���A�QA�p�A�Q�A��A\�
B�CD��<��A  DA�p�A33�Aq=�@
��AR��A���A�,CD�(ܿ�p��  �@�z(A)\oA�p�A33�@{�A���A
�CDq=
@�G�>ף �\�A=
KA���AR��A�Q�@)\�A{�DDR��@
��@�̬@��@��]A33�Aף�Aff�A�CA�8FDq=�@�z A=
CA�'Aף�@��A�µA33�A�zBf�ID)\_Aq=�A��A33�A�p�A
׫A��BR�B�p$BqFDq=f��(ܾ�z�@��A�(<Aף AH��@{�A�Q�A��FD�zD@�5���(@R��@R�JA�GmA��QA\�"Aף�Aq�GD�@  �@�z��q=�@q=>A�̆A{�A�Q�A�peAH�ED=
��Q���Qؿף������<@��@�!A��AR�ED
�#��G��̬�����ף0�
�@ף�@H�A��GDffA�(A)\�=  �@q=�@q=���z�@)\?A)\�A�LHD�µ?�1AH�&AR��?�p�@
�A�̴���A{VA)�ID��@��@�z�AR�vA)\�@\�*A�[A��(���TAB`MD%uA�n�A�ʭA���A;��AZ�A���AZ�A�vjA��QD�̈A�B��B�KB`�ABV?Bu�B�K,Bu�8BR�OD+��A��A�p�A���A  %B�p"B)\�AffB%&PD㥛?�A���x1A}?�A+�A�CB/�)B�M'BD�BH�OD^���)\>!����A Aף�A\��A���A\�%B  #B{�PD33S@V@�(\@-r�VUA=
�A�zB�(B��2B
�NDR���q=z�P����Gq�m�/�Zd�@)\�A�G�Aף�A�SND33��(�R����&��q=��9�T���s@���AH��A�bMD�Gq�q=���zT���`�0��p�����t=��uAT�KD����� 0���T��$��w���Z������j���v����IDNb��  \��(��\���q=��
׽��r��R������{�JD��a@�>��#�
�_��Q��  ������?5���z��P�KD��X@B`�@;��>����ˡ)��nN�`���~�������JD�\���u�{^@!�B��z$���`��̂��z��{��דJD33s�j����G���G!@�|��3�  p�ff��{���hJD�+�)\Ͽ-��=
׿���?/��ff>�R�z����KD��1@=
@�z�?�n��̌?�(�@`����q=N�=�HD�(�)\��������	���@�\�
��(���E:�=
���FDP��#ۑ�d;w�����u����(��V��ˡY�y��)LHDb�@�(���3�33�����!��IX�{"�33���BKD��=A�ЖA{&AR�?��Y@R�.@
��?�տ�(�?�ID����Nb�@X9@A!�b@����1\��t��#ۡ���{�FD�2���
�����a?H����̊��i�
�s����3�CD�Q@�L7���������?52�H������R���{��f�AD33��¡��������R���9����z���G��(iBD333@�̬�)\��j��\���Q���M��{��{�ED�zHA�GuA�(�@�z��-��H���(0��A���G���ED=
W>
�KAףxAH��@����|�33����,�os�3�ED{����ffFA33sA  �@ף��L7�����q=2��{CD���)\�  ����@\��@)\��q=R�-��H����WFD=
7A�zT@R�>@�(L@�{A�(�A�%A��ٿ�OM�ףCD��,��G!?���\���
�����@R��@H����(H�z?D+������ ���������� ����;���������GD��B�z�A  �@��A�A�A=
A�±A�(�ALHD#�y>��B�n�A���@�x�A%!A��A��AF��A��ID�&�@�z A��'BR��A�zhA���A�̎A{�A�A]KD^��@�pUAXYA�(>B�&�A��A'1�Ad;�A��AfFJD�S�����?��A�Aw�,B�Q�A�{A)\�Aff�A�ZHD����D�@�)\��+�?ff�?%BH�A�� A��A��LD��A)\AZd�@\�2Ah��A��A��TB  B��A{$ND�Q�@33�A�wA#�1A)\�A��A���A��jB=
(B�MD�z��)\�?{�A�G5Ad;�@�zHA+��A�z�AbZB
gODH�A�G�@��(A��A{�A}?�A��A���A���AnSDH�A�Q�A33�A�G�A331B�zBbB�GB�l6B3cWD�G}A��A�z$B��B��)B�pB��QBNb@B��VB�NVDq=���(8A���A33BףB�Bq=_B�@B�/BRhZD33�A�GAA�G�A{0B��TBq=DB�GZB�k�B\�B��YD��	�  dAH�A{�A�z'B33LBף;B�QB��B�\D  A33�@  �A�p�A=

B�zJB33oBף^B�tB�a[D��,����@��y@ff�A�A�z�A�?BffdB
�SBf�]D�A
��@��yA�WA���Aff�A�%B��eB
W�B��dD�Q�A�pBףBף1B=
)BףjB)\YB
W�B\��B��aD�Q@��(�AR��A��A\�B���A\�:B�G)B��hB�KcD��@R���ף�A33�A���A��B33B��OB�>B�!cDff&���@����p�A  �Aff�A33B��B33MB�bD�ǿ�p�ffF@R�����A��A���A��B)\B�Z^Dff��H��{��33g������z@q=>A=
A��A��\D�(���p���������ף��ff���Q�@
�#@{TZD���̀����
�¤p�ff���G*��z\�R���HQ\Dff�@\�¿ff�����{���G���̴��z
�\����YD)\O��Q���g�H��ף�H� ��z#�q=��Q>¤�YD��@�((�
�#��z@��G��
��{��¤p���YD�~�>�:@�� �+���&9������}?���R�XDˡe��QH�{.�q=Z�{��\�r��Q��)\�#�f�VD�z�T�A�\�:�)\�)\���c����\����z?���VD��?�����I8���0���	�\�����Y�R��������(XD{�@�G�@)\?��~��
���H�j�=
��H�
�33����[D�pqAq=�A=
�A��AA'1A�AR�6A�ſ��@�^D
�'Aף�A�(�A���AR��A�A��A�G�A33A��[D\�&�
ף=R�rAH�A��AH�BA�x	A��A  8A��YD\������  ��H��@�zDA{NA33�@oC?�(�? @[Dף�@
�3��S�R�.���EAff�A33�A��A%�@��ZDH���
�C@ף��H�r�{��ff&A�puA=
A��@ �]D)\3A  A�QdA{�@{~�ף�@H�Aff�A33�A�'aD��eAף�A���A��A�z�Aff&A��A��B�#B�+aD��u=H�fA��A�p�A���A���A)\'A���A�(B\/]D=
�{~��G��33A��@�(LA�@)\���Q�@�;\D
�s�  �����q=���z�@�{@33A)\�?ף��rYDq=2�33o����ף�����  ��R����(��Q �͌XD��e��k��Q����	��	�ff���p���,������iZDff�@=
w@�����p1�q=��������I��뱿ffV�{D[DH�Z@��-A���@=
w��p��H��ff��H����@�\D33S@=
�@R�bA�G)A)\�
׋��z��  ���(��R�]D�Q�@��,A�cA�p�AR��Aff�@��H@��L�
�K�ד\Dq=���Q�?��@\�
AH�A�QHAף�?��������]D
׃@�������@��A�zLA
סA��A  �@�Q�?\__D���@H�2A��@��QA)\�AR��A�Q�A���A��HA�EbD��9A�G�Aq=�A��A���A�(�A��A\�B33B\ObD��>  <A�z�A�p�AH�A���A)\�AR��A�(B\bD  @?fff?  HA�z�A�p�AH�A���A)\�A)\B\fD  dA  pAffrA  �Aq=BR�B�pB�zB�,B �fDף0@{�A{�A�G�A{�A�GB��"B�zB�*B),gD�Q�?ff�@���A���A�̜A���A=
B�)Bq=B��eD�����Y���!�T�YAT�eA�IhA���AF�B'1B3�eD�l���z��ff���Qؿ��HA��TA)\WA�z�A�z B��cD
������(`��E����{�@{�@H�@��AF&`D�Oa�����{��j���L7���"���E��E
�;��ͬcDˡaA
ף<33��+��
�_���D�ף�R��@R��@3�aD�����v�@�(��  l���|������������(���aDfff��������@����ffz������Q���̠�R�����]D33s��̀�  ��5^�
׷������C�����33�fV]Dff�ff������������3�ף�����1	�)\�H�\D�E��­�=
��q=���p��}?e��G����Z��^Dff
Aq=�@{N@�?�{N�q=��-���{��=
�� �]DH�J�)\�@��@��L=ffr�ff������h���p��bD�A��TA���A���A�(�A���?=
�>)\��m��@{�bDff@\��AffzAff�A�©A���AH�j@�G1@�(�� @bD��ȿ�G?  �A�GaA
׵A33�Aff�Aff@���?{$`DH��   ������A�̴@��dA�3A��A\�����^D�p����e�R�~��Y���9@q=��{A���@��<@=*_Dף @q=z��pE�\�^���8���@R��?q=&A���@=�^D  ���u�����pi��G����\�q=*@\��q=A�`]D�̜������z����0�����z�����)\���� P\D�Q��\��\�6�ff��u�  ��\����·�  ���sYD=
7�33{��̤��̶�R���{�����=
�ף	�=�VDff"�R�������  ��  ���ף�)\5�ף;�:VD� 0��nN�j��������������5^@iXD;�A��@�����y�H��{��{��  ��)\���ZXD�k�'1A�Q�@�z���G}�R�����������
���UYDq=z@�k@��FAR�A����R�>��p��ף��ף��qmZD�(�@ףA�� Aj�A��`A��y@�G����<�����q�YD  `��Ga?�G�@��@��TA��(A���>ף0���t���XD�zt�q=���(<��Qx?�p=?F�A)\�@H�Z���m���TD����z���z���p���Qt�  x��A���(��G��3#TD��̿R����G���G��q=�������̈�^����1���UD�(�@���@)\/��zl�q=��ff^�
���#�/����SD�z�33ӿ  P�����z���z���p���(��  ��RWD�WAff�@�G=A�#A�Q���G��GU�33��G����YDb4A;��Aˡ�A��A;߫A�Χ@�E�?/���t@��[Dˡ�@�p�AףB=
�A{�A�G�AR�RA��A33�@�[D
ף<�E�@���AR�B33�Aq=�A�p�A=
SA��Au^D��Am�A)\�AZd�A��$B�~BB��B�O�A�*^D-�?�pA��A�I�A�Q�A{'B��B�z B{B)`Dף�@-AH�A=
�A�r�Aq=B�(EB=
#B\�>BD_DbH����@�Q�@w�WAbXA�p�Aj�B�8B7�BZD^Dw��m������>�ʁ?��A� A�x�A��A�(B��]D�l��^���H����h�+���H�A33A�A=
�A
�]D33ӿ�O=�+����G!�
�#��������@���@��gA{�aD��A��xA1dA�$A�(�@ffjA�AtA��A{�A��`D���zDA{*A�&A�n�@��@�AB`%A�z�A
�aD=
�@=
W�  �A��uA�`Aj� A�p�@=
gA`�pA �]DH��q=>�\�����>�G��Z$�J��=
�H�
�õ]D{�>�(����8�
׃�H�:?�k����+������N[D����Q�=
���G��R���{��z(�sh=�X}��[Dףp?R�
��G�����¡�33��{���p�5^.��[D
��=
W�33'���!�����  ���p������5�=�ZD����ף�����H�.��p)�����
׳��G��33#�B�]D�1A��)AXANbA��'>%?ـ�-6�+���\�]Dj|?�GAA��9A�A�(,A33�?R��?��q�ff&��[Dff>���.��Q8>�������둿  ,�\�&��(��fZD)\o�q=z��vj�
�c��G��q=���(��
�g�ffb�\oXD�����%�  �����R�"�ff*�H�F�
�7��̨��jUD�A��p��)\���G	�V���µ�  ���z��
�UD�(@{����
ס�� ��&��ff��q=���z��\YD{FA�iA   @���33��  ������p��ff�\�YD  p@=
�A\��A  �@�Q��33��  ��X9p�H�z�
[D
׃@
��@  �A��A��%Aףp@
ף<{>��M.���\D�̼@�Q A�Q\A33�AR��A�(�A\�A�p�@)\��ہ`D�&�AZ�A�O�A�O�A�(B�o1B�'B�n�A��A�2^D�����@��JA�n�A�n�Aj�B�~B�n�AP��A=�|Dd;�B��BffC�LC�kC�+C͌Cq�C��C3�zD����B���B{��BHa�B\OC\C�pCH�C �{D��l@���yi�B'��B���B��C�C��
C�#C�D�G�AH��A  �A��C�aCffC�LC�kC�+#C�1�D���ff�A  �Aq=bA��C�
�B=�C�pC\�C��}D��%����33Aff>A\�r@���B�W�B�0C
C �}D��L���(��G��   A33;A��e@yi�B'��Bq�C���D��A�Q�A=
+A\��@���A��Bף�A-rC6
Cq̀Dq=��R�vA�sA
כ@�p}�)\�A���A{�AB�C�ҀD
�#>����GyA{vA���@�zT�ף�Aq=�A)\�AHa�D\�b��QX�33�ף@A�p=AR��?
׋��Q�A��A�C�D{n�=
�����{"���1A\�.A)\?����H�AHaDH��ף�����ff��k�ף�@q=�@����q=� �D�p�?=
W��G��\����p��
�W�  �@���@333���{D��l��Y��G��R���=
���³��Q������  ���~D�GA=
������G!��(0���h�q=f����
׃?å|D{����(@\�B�H�.��Qx��������ף��33����|D�Q�?  ���u@�/�
���Ge��(t�ff�����
�D��A=
�A��A�(�A�A33/A��@�­@���?\g�Dף0@���A��A���Aq=�A�GA)\[A��A=
A�1�D��տ��?q=�A�·Aq=�AH��A��,Aף@Aff�@��Dף ��k��k��(�A��A�QxA�̼A��A�z Af&�D33������z �ף��R�VA��iAR�A  �A��@דD��8�
��  4�R�N�\�"��z(A�;A���@��eA=B�D�(<A��A�G!@\�?�z�����?�Q�A
׻A�Q�A{��D���GA{�@{�>��տ��U��z�H�Aff�Aý�Dף`A��=A���A
ץA{fA��EA33+A)\WA��BM�D�GA�(�AR��AffB
�BH��A���A�p�A��A��Dff�@�p�A���A�Q�A33#BףB�z�Aff�A=
�A�r�D�Ga�=
�?\�ZA���A�(�A�B\�	B�Q�Aq=�A\σD���{
��{���AR��A�G�A� Bq=�A�p�AR��D�(��\�������̤�\��@���A��A���A�z�A�E�D�p�@��l@�z�����k�  DA�Q�AH��A�zB�*�D=
W�\��@=
7@  �ף������\�6A���A�(�A׻�D�zHA=
;AH�Aq=vA�z$A�Q�@)\7A��A��B{ĆD�QAff�A��A=
�A�G�Aff�A�zpA
םA
� B��Dף�@ףlA\��A
��A33�A�p�A\��Aff�A  �AH��D
�����Y��p�@���AH�Aq=�A�z�A���AH�bA���D\���33k�H���Q�=��IA�z<A���A�wA��%AH��DR��  �������̊��G���\@=
'@���@���@\�D=
CA�GA@���H�:����=
G@q=zA��lA�A���D�G��A��̽������l��z�
�#��QHAH�:A ��Dq=nA�Q<A��AףlAR��@�Q�=��@{nA�G�A
ǇD�(�?H�A
�SA�p�A{�AH�A��?�GA�̂Aד�DfffA��}A{�A��A�QB�G�Aף�A)\A
׳A3c�D�gA=
�A���A��4B�z(Bq=YB\�4Bq=B��A���DףP��3A���AR��A��'B�pB33LB�'B33
B���D
׳@=
@R��A���A
�Bff>B��1B�bB  >B�̋Dףp?���@33S@q=�AR� B��B�(BB�5B�pfB���D=
�   ��Q��z���pA��A��AffB��B��D  4A
�@  P@��A��@R��A��	B
�BffKB�#�DH���q=j@����H��
ף?�p��  HA33�A���A  �D���)\{�R���ffV�)\G�H�����!��(�@q=�A=��D�EAff�@��X���@q=��)\�)\�@{@���A�[�D�Q�A�pB���A33�A33�A��A33�A�(�A{�A\��D�@�G�A��BH��A�(�A�(�Aף�A�(�A��A�G�D
�/���	�R�NA���A���A�zAq=�A�p=A�zLA�E�D��u���0�H�
���MA�p�A�z�A�A�¥A�z<A\_�Dff���Q��  ��{~���@
׏A��A�G@�QXAͬ�D)\�A�A�G�AR�Aף(A�Q B��1B�B��Aד�D�G���A��A=
�A�z�@�(Aff�A�z.B  B
��D33�@q=�@��B�Q�A
��A
�oAH�A��BH�LB�ЏD{���z�?ף�?ff�A�̢A�Q�A��AR�:A
�Bfv�DR�RA)\�@�GqA��dAH�"B{B
�B�³AR��A���D���?
�gA��A33�A��yA�((B)\B�B�Q�A3�D�p��fff��A�Qx@�7A33+A�zB)\�AH��A�z�D{�$��z���\���Q��\�����Y?33��\��D�̴��zk��(�H�y�33E���[�\�=��@�  ��R`�Dq=�A��Q�)\�=
(�"��(��
��H������q]�DR���{�@�G��R�T�ffh¸c¤p.�33E���&�{�D���A{FAףB��8A���=
���z��q=r�ף��
��D�Q6B��B
�gB�z�B\�dB���A33A�(�A��Aí�D�(>�H���{�AR�&A�� B��A)\��R����(��E�D)\�@q=+®G1@���AffrAR�B�GeA���H���f�D�G�@�QA{��@�B��AH�'B���A33�����D{�Aff�Aq=�A{��\��A�(TB��B��jB�B.�D
׋A��B�B=
0B��a�33(B=
�BR�YB�k�BÝ�D�Q���OA
��A{B  Bף ��(B�B�GB�ŔD  �A
�KA���A��;B{PB  cB)\A�([B��BV�D)\_��(\A  A
��A��-B�BB=
UB=
�@33MB���D)\SA�A���A��A��B��bB��vB���B�p�A�Y�D��9@H�A��IA���AH��A)\+BffnB�G�Bq��B)\�D�GA�/A��A���A��B��B�KB)\�B�p�B�J�Dף����(@ffzA\�BA�G�A33�A�)B\�lB���D\��?ף����?R��@)\�AH�^A�p�A)\�A��0BHy�D
ף����?H����p}?��x@�̈A��YAH��A���AR��D�AffAR�"Aף�?)\A��IA\��Aף�A�QB ��D��@��Aq=~A�G�A��A���A�̠Aq=B\��Aq��D{.>��@=
�A�z�Aף�AףA���A�(�A��B�f�D��u�q=J�ff�@R�vA��qA���A�GA�G�A�z�A���D
�c�ף��33����H@��=Aף8A��TAף�@��MAV�D)\O��(��
׋��z�����̌��µ�33�>=
�ݗD�CA�p=������Q��H����@��1A��,A�IARH�DffV@�yA=
'@33s��z��R�޿  �@�gAffbA�s�D�̬?ff�@)\�A�p}@���>)\��Ǿ33�@�}AHٗDq=��{^�������AA�(\��p���(��R�����@=r�D{N�ף �{����U�{A\����z����	�H�����D�G9���l������(��R�n���,�\�z����p����D���@ף�������D�)\/������@���\�>�졔D���H�.�{��
���ff����������{Z�R������D{��  
����{���'�q=;�
�5¤p(¸��3�Dff�A\�B@33s��z�������{���G���z��Rh�D�G�A
�B���A�p}@�Q4AR�����a����  �� �D�(@����@���A=
#A�� ��p=��E�ףx�H����D�����a���@H��A��Aq="��5�\�f�=
��H�D��a�����������
�#Aף��{������q=��ý�DH�����®G%��QU���m�=
���zE�q=(�
אDq=J?\�����¸"��(R��®Ga�R����QB�
�D  ������GL�Ha��\���{����p��Q#�)\d��H�D�����1�.���¥¤������ף��{j���D
׻@�=��G¸�\�{���33��R8��f��RP�D�z���k>���
�0��-�\���L���z�� ���{܍D=
�A
�kAH�AH�:@ף���Q����@�{y�R���3�D=
W>R��A33oA\��A�QH@����ף����?�q=x� �Df����Ĥ�Ě)	�����ą�ď�� PD  ��fv��hĤPĚ���A���^ą+�R(D
ד������Ĥ��Rx�H�Ěi����ÅĤ`D=
�A�(DA
�_��e�RX� @����H1�q�ď�D�zA�G�A�Q�AR�����f����
G
�\��)�D�̢��-����@�Q�?�z��=���{��q]ĤpDH��q=��  ���7��������U�RH� 0��LD{�@R���R�����D�q=�@��L�ff���y�)l��,	D  `@=
/Aף @R�������A��\@ff����Ě�D33sA���A��A��A��H��z�@�(�A33�A33����D�z�����{�\��@ffV�����R�b��p-@�z����D{~�q=���G�\���{�?q=��)\������G�� `D�(�@q=
@33����L����>��@�Q���Q���(@��`D���{�\������=
3�{��  ��H��q=��)�D�zD���0���ff�����(d��(,��zT�  D�f�D�p5�\�f�33���(�����33#������̰�\�j���D����)\7��zh��(�����H���#����±�D��UA�SA���?=
��R��\�B��Q��\�����E�R�D����ff>A�z<A�G�>�Q(���)����)\��(���D��i@��@��xAH�vAH�@\��?=
��=
W�����wD����G���p�q=.A�Q,A�����h�{:�  ���DH�?��L�ff�>  ����EA�CA
�c?��R�"�{�D�(L���4�)\���D��(\���Ѿ����=�=
o� �Dq=��ף����}�q=��=
��ף��)\��33������
DH�
@���\�r�33[�H��)\k��G��
�3��;���Dff�@��	A���@)\�  ��\�>��(�)\���h@��D��<A{�A�p�A��}AffF@��@��̽33c@ff@�bD=
G@R�nA���A�Q�A�AR��@�p�@ף@@��@=�D�(��ף���G�@
�cA�G�A�p=A�k�q=
?
׃��GD�G��R�F����   @33A��1A���@R���  ��3D��Q����
�S�{"�=
�?{A��$A)\�@�����D�z(A)\A�p�@�p-����>)\CA�G�Aף�A{�A�D��?�pAA�Q4A)\�@33��H��?�Q\A�¡A��A�l	D��A��-A33�Aף�AףtA\�A�Q4Aף�Aq=�A��
D  �@��pA���A33�Aף�A�Q�A\�^A�(�Aף�A3�D��A��aA�G�A���A  BR�
B��A{�A���A��D�1A)\�A)\�A��	B�(B�G:B  7B
�!B�QB͜D�p�����@  pA  �A�z�A���A��(B�Q%B�(BףD�G�=����Q�@��qAH�A)\�A
��A=
)B��%B��D��?���?{N�33�@���A���A{�A�GBff-B�D����=
������z(�q=
?q=Aq=jA���A{�A��
D�G����ff��=
�H��ף�ףp���@��aA�p
D  @��G)����ff�����H��ף ��Qؿ��@�;DR�2AR�&A=
?=
���(��ף��=
�ף�?�A�D�̌?�QDA�Q8A�Q�?
׳������p���p�R�@��Dff�@���@�A�A�z�@�k�{.>)\�>�z��3D���G@�W@ףhAף\A�p}@ffV�ף���	� �D��L�R�N��(�?�z$@
�[A
�OAq=J@�̄�
�C� �D   �����3������­���@��@H��ffB�fD��Y?����ff��{&��µ�\���H��@H�@)\�3
D�� �ff��33s�  ���p���[�{J�Hế�p�\�	D�׿����(�{���z�����ףv�=
e��GI�=*DR� A��AR��>�z�?H����z���!�
׫�ף��\?D\��@  fA=
KA�z�@��@ף �
�S�������H�D
��?��@�z~A�cA�p�@ף�@H�z�
��R�����Dף�@��A{ZAff�A��A=
_AףlA�G�@��@�"D  �ף�@���@{>Aff�A��A=
CAףPA�G�@�{Dq=VAq=:A�G�A��A�(�A�BB�Bף�A�p�AH1D�z���CA�'A  �Aq=�AH��A��BH�B)\�A)lD�k?��u�ffRAff6A)\�A���Aq=�A�LB\�B�*D\�����̽�G��{BA{&A33�A�p�A{�AR8B\�Dq=�@���@=
�@��@���A���A���A  �Aף�A�D��5���@H�@�Q�@33�@��A��A{�A�Q�AED33@��?\�A�z�@��A���@�Q�A�Q�A�z�A��D�z�
ף�H�:�H��@q=�@��@\��@�A�A��D��E�H�j�{F��pQ�ף���G��
���������@
�DR��@�������p��{�������z���z$��D�zT����?�+�ףP�
�+�337��(���̴�)\����D��@\�R@  A��������(��H���p�?��u?��D��@�cAff.A��}A  `@=
�?R�^@�G1@�pA�CD������@�QPA33A\�jA33@
ף���@���?��D�(���h�{~@�G)A�Q�@�CA)\��p�
�#�{4D���
�C�=
W�\����G?\�"�H�@=
��(D�
GD�Q�A
�gA��@A��-A��A\��A  �A��A��eA)�D�p�����Aq=�@�(�@�@ף<A33�A�GqA�Q�A{�D�(�@�ſ  �A33OA�((A��A)\�Aq=�A��A)LD)\�?  �@
ף=���A�iA{BAH�.A�Q�A33�A��D��)\�q=�@ff���Aq=FA33A  AH�A�D���������Q������{�AR��@ף�@�zt@H�Dףp?)\��p�������z?H������AffAR��@�|Dq=���(�����
�3�������\�2�{RA�U@)�DR��?�e���(�q=��  �{�  @�R����iA3�Dq=~�fff���q=���������(������z��ffDff��R����̢�)\��
����G��R�������R���H�DR�
A{.@R�R�H�:�  ����|����)\��ff���HD{��뾸��ff���z��=
���������33 �\�D�̬?����
�c?����������q=��R����(����D�+�q=��)\'������=
���������(��)|D�(�@��\@���@\���H�@{ο  ��{��ף�� PDף0�{�@ף0@��@ף����i@33���������D�(��q=��R��>��\���ff"��p������\����#D=
GA���@
��@  LA�!AR�6A\�@)\/A�Q�@�)D�Q�=�zHA���@R��@�pMA\�"A�(8A�Q@��0Aq�Dףp���Y��p9AR��@ף�@ff>A�A�)A�Q�?�,D
�A�� Aq=Aף�A33wA�(lA��A��A�z�A
GD��@ffVA)\GA��HA���AH�A)\�Aff�A���A=�D��ٿ�p-@33;A�(,A��-A�Q�A�G�A�A�̼A�D)\?�{��)\��)\Aף�@��@ff�AR�rA�gA�"D���=�7�q=��
�#��GA�z�@)\�@)\�AףtA{�D�(�����H���G���Q���@�p�@�Q�@���A�D����p�������G��
��\�����@{^@
�c@ �D
ף�q=��ף�R��\�F���a�33���5�33ӿ{D�+�����  $��C���A��pq��Q��{F���X�
�D\�@
�#��Q��)\�H�"��� ���P�  l��p%�R(Dף�@��A�(�@\�?�Q��q=z�\�r�������HaD
�c?��@33Aף�@33�?ff���GA���9��z���HD�g@�Q�@�z,A�MAq="Aף�@�z$@��?�Q8?�qD�z�@�(AffAR�vA��A�zlA\�AR��@��@RhDffv@
�A��AA  PA�(�A�z�A=
�A�(XA��0A�D\���������@��U@��Aff:A�Affv@�ZD���?)\����)\�>��y@�G�@��0A��QAR�&A
�D
���(\��Q(��p�������?R�@  Aף,A�9DR����Q�q=��
ׅ�{N�
��
ד�R�n�\�"@�hDףP����z<��G%�������  8��(������1D�z�@�Qx@���z���(L���M�  �����p=��RD�Q�@ff:Aq=A�@�����?�p������>HQDR�~@
�A{zA��EA��@ffv@���@���\���YD{A��AA���A{�A  �AףdA�?AH�VA�Gq@HQD{�    R�~@
�A{zA��EA��@ffv@���@��DH����c�H���=
���@ףA���@)\�=)\��D��@33�?)\��33�?�(�@q=Aq=�A�QXA���@ PDR�R��½��Q@�33���Q@�ף ���a�=
g@33�>��D��)@�Q(���Q����  ������z���Ga��Q�@��D�Q��
�#@��)��W�)\�R���)\�)\���Qx�=D��?\�?q=J@�( ��G1�����������(��R�D�z�H���Q�
׳��Q����8�����(�����R8D  �?��������ף��
׋��Q����$�����(�� �D
׫@
��@q=
��ǿ33ӿ  �?R�B����Q0���D
��?���@ffA�G!�����������!@q=*���Y�f�DH�Z�ffB�����������d�)\[���\�ff2�\���\�Dq=A�G��ף �=
7@��@�p��q=������G��׳DH�N�\�R���=
w��!����̌�  ��R���)�
D��y�ff���G��)\��z��������H���z� 0D=
'��z����������,�  &�����ף.�),	D�(|@  ������33®G��)\�q=� �����.DR���ף��)\g�ף��=
,��z�33<�{6� ��:D
�s��Q<�\����(��\�®G;�R�¤pK��QE��D33_�{������q=��H� �)\?�{s��L¸���RD\�A�G������z����E��z��R�¤pM�H�&��D�̴@��pA{�?��,�\�*�=
���G����
�6�)�D��@��$A�Aף�@H��?  ��
��  H������i
D)\/A��yA�(�A�p�A
ׅAR�NAR��@ffA�ſ�9	D  ��R��@��-A�(�A�p�A�?AR�A=
W>�̄@Rx	DH�z?�Gq�{�@��=A  �A�G�A)\OAffA�Q�?)�D��Aף A�G�@  �A�G�A�z�AH�B�(�A��A͜DH����	A��A���@�(|A)\�A\��A��Bq=�AHD���?q=�?q=&A��5A
��@ף�A��A��A33B��D33�@R�AH�A��A�A�cA�p�AR��A��BH�D33��  `@q=�@\��@q=^A��mA��!Aף�A���A)D=
�>�k�H�z@��@  �@��dAףtAף(A  �A��
D
��q=��R�>�q=��   �)\?�q=�@���@33@)�	D�Q��  H��GA��p���G	��Q��  ����?\�2@{�	D�>�(����E�33?�ff��33��(��
����Q�?��D��@�G�@��@R����G��q=�\���Ga���5��RD
�C@�A��AH�@��9�R��\���\��?��5@ �D\�B�
ף<���@���@33@{��ף������G�)D��33����
ד@  �@H��>  ��\���H�6�)�
D  ��
�C�33��\�B��W@  `@\�B�  ��G	���D��������9�ffj��p9��¥��������ff��� D�Q�H�*�H�>�
�[�q=���[���������=
7���D�(,@)\?���
����0��pa��z0�
ד������D�(\�  @��Q8�H�6�H�J�
�g�q=���g��� ��xD�Q@���ף�?  @������(���E�ffv��pE���D��?{n@)\�>{>@=
W?R���)\��Q,���\��N	D  �?��U@=
�@��@=
�@��%@R���R����Q���D  ��  �>��?{~@�?{N@��?R���)\�E	D�̬?������?�(L@q=�@ף�?q=�@�(@�����DH���G��G�\�������������������åD���
�'�q=�q=*�q=������p����	��p��>D)\Ͽ�+���A��(,��(D��((�R���G���#��D��@)\@�;@  ���̤������̜�
�S�=
��D  h�q=��((��9�  ��33��33��33���z����D��)@��=�������R���̒�  ��  ��  ��
D�(�@�z A=
���(\�R��\�b��O���9���Q��D���(,�
�#�ףh�H����(���9��Q�����
Dq=�@  ��ף�?���@�'����)\���G�����`D��)�������i���q=��\���)\w�ף������� D�p���(��33C��(��q=n�
�C�����=
��  ��{�C�p���̺����=
����®G�)\���1�q=9�CףAq=���p��G��R����G��q=��=
���®� DR�A��A�(���z�����R�V����H��)\W��D�p�@�pYA=
�Aff�@�(���A�   �\���=
+�3D�(DA�p�A���A\�B��Aף,A\�B>�Q�@
�s�H�D=
�@
וA33�A�GB�p"B�p�A{�A��@�+A�;D���(�@��A�z�A
��A{BR��AR�vAq=�@HaDH�@  0@�A
׫A33�A�GB�p-B�p�A{�A�SDR���\��>Hế�Q�@�(�A��AH��A��B�«A��D������33ӿ)\_�R�>@
�sA�G�Aף�A�zB=�D
ד�������-�ף�������ѿ��)A�Q�A��A3cD{��)\���Q���?��(����  0��(AH�nAq�D�p!�333��}��̎��z�����z���pM��z��Q�C�M��G���(�����)\�����Q��� ®G��3��CR�����d�33��{��=
��ף ��z�q=���z�{t�C�G�?q=�>��H����  ������33���p
��(��)\�C{�����p�����ף�����q=
�)\�33#®��Cq=j�����p��G�33�����ff�H��  !�f��C��@q=�?���ffV�H���G��  ��H������
��C�(�@��AR��@H�z��G?��5��zX�����
���H�C�����p@ff�@\�R@��9��둿�Q(�33w��Q��=:�Cq=�@�̐@�z
A�QVA��AH�b@�©@�(t@{�R��C
���½@�Q�@q=A{NA�A��A@�G�@33S@�D33'A��A=
�A)\gAR��Aף�A)\�A�WA
�sA3SD�G�AH��A���A�(B���A  B�� B�QB��A�|D�̔@�z�A=
 B���A��B{B�� B\�3B��$B.D)\���z�ף�Aq=�A��A
�B�Q�A�BףB)LD�zx�{��H����L?  4A��+A�p�A�(tA��A� D�(��Q���(������R���)\@���?R�A  �@�N D��L?R����������\������\�R@��1@�A3cD\��A���AH�A�zT?�̌�  �>�G�AH��A���A�D33���A�(�A�(0Aף��  �33����<A{�A3SD�̌?  ��\��A���A��AAH�Z�ff
�  ��\�NAf�Dff��33��337���MAR�ZA��@��)���}�333�uDR�n@{^���=
���̄A33�Aq=
A�z����A� xD�p�@���@33?ff�?33k��(�A\��A��JA{6��QD=
������K@ף��H�:��Q�ff�A�̆A�pA=
D{.A��@�G%A��`A��@R��@=
'@�p�A
��A �Dq=
��A  �@R�Aff>Aff�@���@ff�>�(�A��D�( A33�@
ׅA�(BA�p�A�G�A)\KA��\A)\A=�	D�̤@\�RA  0A=
�A�G�Aף�A�z�AH�A��A�@D����  ���Q�@33�@�{A�(2AH�rA�G�A)\;A��D=
��(�=
W��̔@)\@��UAffA�MAff�A͜D33s@�Q�?R����G�>33A�G�@)\�A33IA���A3sD�̔���Y��pM����R���33s@���?�QHA���@ �D����337������( ��G���(0�  @����=
�@��D��i@��I�����  ������(H����ff&?�G���DH�
A�GEA���@�Ga?�@�z@�u����?�GA��D)\?A��A�Q�A��A�pMA��A�zdA{A�zTAR�D��U@��tA
׿A=
�Aף�A�p�A
ןA���A�7AH�Dq=FA�{A��A�zB{ BH�B\��A�zB{�A�D��q@�Q�A=
�AR��A{� B./B��BHaB{�B�D{�
�S@33{A�Q�A  �AR�B�Q-B�B=
�A�[DR�F@��(@�G�@�p�A�(�A�kB�#+Bq�9B=�B�9D\���­�ף�����Q$A��YA\��A  	B��Bf�D���?R���)\��q=���k��7A��lA�(�A��Bf�D  d���P����
ד�\���R�r���1�)\?�QHA)�
D����G������������
׸�ף��{��fff��JD)\�@���?H�N��;�����G��  ����]�Hế @
DR���(,�33���̨�33���z��ף��)\���(���hDq=
Aף�>ff�@��?)\G��(4�)\�����q=��
gD������@
�c����?)\����ףt������¥�=�D��Aף�@\�RA��@�'A���@{�����fft���D)\����@��@R�*A   @)\�@�zt@H�&����<D
ף?HᚿH��@
�S@33?A��q@�(A33�@ff��WD=
�>���?q=J��Q�@R�n@��EAff�@H�Aף�@�bDR�BA�pIA��]A{6A�p�Aff~A�Q�A���A�̮A �Dף(�ף�?33@�U@=
W?q=A��@  `A\��@��D=
W�ff^��pݿ��������G!����@   @q=*A3cDH���33����q=������
�����	���u�H���HD�pe@�Q ����q=������(t�q="��̠���a@ͬ
D����ff6�=
���D�R���R�*�  $���)\7�\_	DR����Q:��� �ffb�{��ff��{~�)\w�H�b��SDH��q=V��������ף���������z�������D��(@�G��  ,��z����Y����ff��)\�)\��
7D��h?33c@�(���p�33��=
K�q=�����R���5	DR�~@�z�@���@ff&����R�D�)\���l��G��H	D��Q�q=J@q=�@R��@�(������
�Q��z���y��"	D�?�����k@H�@)\�@33s�����I��(��(	D���=R�?��L���q@{�@\��@��Y�������G���D����(�
�����\����p��
�ÿ)\�R�n��BD�G�?33��  ��)\�������zt�q=:�q=��33�H�DR�@)\o@
ף�ף��  ��q=������(ܾ�p@\/D�z,�����G��ff~���|��zt��̀���A�)\3��!Dף<A�G�?)\_@  �@����Q��)\_��������3�D�pͿ��"A�Q��G�?�GI@H�����=
���G���K	D�(�@�̔@��A��@q=AffAq=
?
�#?�z�?q]D�zA\�hAH�NA���A=
_A)\�A�p�A�AR�A=jD���A
��AH��A=
�A�#B��A�zB�B�(�A��D�G��\�
A��A\��AR��A�B�̴Aף�AR��A�D  �ף���@=
sA\��AR��A=
�A�̦Aף�A�XDR�>�)\���QD��{@)\CAR��AH�A33�A���AH1	D��I���y��̊����=
���Ѿ=
�@��@q=�A=�	D��@�#�)\S�)\o�  �������p�?��A�(�@�5
D=
�?q=�@����z8��zT�\���
ד�q=j@ףA��DR��
��33����������������ףf��Q���DH���{��ף��=
[��z���Q���Q���Q�=
����DR��@ף����,����)\���̚�ף��ף�������<	D��@��]A���@��x��p��Q8>=
G�R�v�)\���qDH�J�)\o@33+A=
/@�����(��)\?���y�R�����D�����#�
�S���i@{����a�H�F�ף ��G��\�D\�B?�G��)\�33#���@�c���U�R�:��z��L	D)\A�'AH�Z@  �>��@��aA���@��h�H���\�D�p-�  �@�Q�@��5?�p�ff�@\�6A�z\@33���D�Q8>��!����@{�@
�c?����(�@�p9A  h@�Q	D=
'@\�2@
ף=ףA��(A  `@���>��@33cA� 
DR�N@H�@ף�@
�S@�QPA�z\A)\�@
�c@�'A�3D���@���@q="A�%A��@\��Aף�A�z0A��@H�
D\��ף @��@33�@�z Aq=�@�zpAף|A
�A)�D\������q=������?333?ף �ff�@)\A �
D���@q=��
�#�R��?=
�@\��@�Q�@���@�(lA��
D)\����@{���Q(����?�̜@�Q�@{�@)\�@{TD���H��)\�33�
�7�{���p}��̬���qM	D��x@=
���G���G@���33��33S�)\��\�"@�|
D��@{
A�k������Q�@���=
7��Q�?�p�@��
D��?\��@�A{�?��?��A�E?\�¿��1@)�	Dfff�ף�R�@
��@)\�
�#�  �@�5�
ף��a	Dq=
��Q���p��
ף>R��@�̔�=
����@�����D����)\���<��'�)\����33+��Q,��pm�=�D����
����z�{R�ף<�����q=J��Q@��pA�f&
D=
'A��A�zD@��h?�(,��̬���X@���@q=ʿ��	D
�#��(�@���@\�?33ӿ  ��q=z��zT?=
�@�D�z$��(����@)\@
��=
��q=��)\��R�޿ �D=
���G�����=
W?��������=
�ףD�33/��7D)\�?ff&��p��)\���p-@��?���q=��R�&�f�D�p�?fff@R�^��(\�  ��{�@�G@�;�H���D�GA��4AH�RA)\A�z�@�e@�Q`A33KA���@HaD��?R�.AffJA�QhA�� A)\�@�p�@��uAף`A��Dq=
@  `@�GQA��lA�p�A)\CAq=A\��@�(�A��D�pm@
׻@R��@�Q�A�(�A��AR�~A��UAף,A{�Dq=��{��ff�?��@�?A33[A�yA��1A�zA�|DR��{���pݿ�(�>\��?��5A�GQA33oA�'A�C
D�z���Q���G%�
���R����G�R��@=
A�� A PD��<�����\�������p���(��H�n�33���g��DHᚿ�QP��G��q=���̺����
׋��������3�DףP@33@�(�ffj��Qt�R���=
���c�{N���D�G�@���@H��@=
����)��3�ff���pE�H�"��gD�(ܿ�z@\��@
׋@{���GE�33O��(����`��	D��@�(,@)\�@
�A�zA��A��p��ף���U��lD�G�H�z��z����̿�z�?ff�>��5�  �������)D��<@�(���Qx��(,��̬?��@��Y@\����T�
D)\7A\�fA\��@
�'A�QA��LA\��A��mA33C@��D��yAף�Aq=�A���AH��A��A�p�A��A
��A�D33GA\��A�B��)B��Bq=B)\B�#B\�0Bq�D�Ga��CA���Aq=B=
)BR�B)\B�zBף"B�qD\��A�̌Aff�A�6B�dB�QpB  OBף`B��YB��.D��B�Q�BH��B���B��CR�C��C
�C  C�5-D������B���B�Q�BR8�B�C�0C�#C\�Cf�-Dף @ף���#�B���B
W�Bq=�Bq=C3�Cf�Cf�-D  ��ף@ף��ף�B�G�B
��Bq��Bq�
C3sC��(D��������H�����f�B=��B��B  �Bq=�BH�#D����Q!��Q"®G�ff3���SB���B.�B{�BH�$D  `@����Q��Q®G
�ff%���aB���B.�B=�$D��?�z�@�q�����R��
���gBff�B�[%D\��?q=J@��@��T����®G�����nB�#&D�QH@�̜@�G�@ףAR�"��Q���Q��q=��q=� `$D�����{�q=
��E�R�.@
׉�ff�ff�)\�.&D=
�@
�#>\�R@��@ff�@33A�( �=
��=
���Q%D�(\���q@��Q����)\�?ף@@�Q�@33W��GR(D�(@A�	Aף|A�A��=A{ZA�QpA�(�A�Q����'D��H���A�­@ffJAH�@�A
�'A{>A{vA�&D�Q��ff�=
G@�����z�@{.��p=@)\�@
��@��%D  ���Q��ff"�{�?q=ʿ�z�@�µ�H��?R�^@3�,D�G�A�G�A33�A{�A�(�Aף�A33B���A���A�,D�?ף�Aף�A\��A�p�A��A  �AH�B�G�A��+DR���\�r����A���AH�A�SA
��A�Q�A{�A�W,D=
7@ff�{n�
��A
��A�Aף�AR��A33�A
�-D��@33�@��(@
�s@���A���A��A\��A�QB�.,D�(��
�#�{@)\/���ȿR��AR��Aף�A=
wA��-D��@)\���p�@���@�z$@)\o@33�A33�A��A��)D��h�  �{j�q=�����
�?��-�R��A�pqA�!D���=
H���2��QH�)\5�)�=�{9�H��
�D)\���zu�)ܗ�q=��  ������̈�R���Ha��D�z���z�����#����®G���̙�{��  ��\�$D{=B�&B)\{A���33	����z	�=
���(��q�&D�A��]BffGB�p�A�Q8�ף���(��33���G��)�$D�Q���L�H�<B�Q&B\�zA�Q��ff	��Q���	�{�!D��=����R�>�ffB��A\�r@�G��H�8�ף#��cD{��=
����¤p��R��A337A�g��G�����=:!D��uAq=:��zl�ff���Gm���Bff�A�Ga?�G
�3CD�����@�Q,�����G�����ף�A��A)\��q=D�p�33����H��
���  �q=����A��-AR� DR�nA\��@
׃��eA�(|���|�ף����}��G�A=� Dף�33gA��@  ����]A���q=��ff��ף���"D�p�@ff�@���A�z`A�p�@�(�AףP@��	�=
��3�#D=
�@q=>AR�6AR��A  �Aq=&A���A)\�@�z��#D�5���?��A�p	A{�AR�vA���@�G�A�̔@{D&D��IA�QA
�_A�G�A��A�pB�(�A�G�A=
BHa&Dff�>��PA�#A=
gAH�A��Aq=B���AH�A��'D���@33�@���A���AR��A{�A�Q�A
�*B�z
B�:%D  ,�33���̄�33A
׳@�pA{�A�Q�A��A��&D���@{��H�?�z�?�(hAH�:Aff~A\��A�̸A�L#D�z\�=
�����E���=�H�:?ff��@ףA�R#D�Q�==
[��(��=
���C��z<���Q?ף ��p@=�#D)\�?H��?�A��Q��{����)�\�"��(@�Ǿ�	 D�(l�q=R���P�ף���(���(������)\���E� Dff~��G���Q��������*����=�{%®G#�
�D��ѿ�Q��33¤p��R����z1�q=�q=D�ף+��LD�piA33OA��<����R���  ��q=���½�H�	� � D���@��A  �A���?\�N�ף4�333�
���)\��\?$D
�oA�Q�A�	B��BR��A�@��l@\�r@
����'D=
kA�p�A��	B�GDBR�=Bq=�A�(�A��A
דA��'D
�#�ffjA��A��	B�DB\�=B���A
ׅA�̒A �'D��ѿ=
׿�(PA  �A33B\�=B  7B���A�pqA �&D  @��z�����( A  �Aff�A\�1B  +B���A\�'D��@R��?��̾�G�  dA���A�(B�BB��;B�)D{�@H�A���@�G�@  �@��AR�	B��B�GWB��*D��@H�>A)\�AR�RA�z8A
�7A�p�A�$BH�7B=*Dף0�R�~@R�A\�VA\�&A�QA�A)\�AףBHA-D��IA��AR��Aq=�A�(�A�(�A=
�AR��A�B{�-D33@\�nAffBA��Aף�A\��A\��A�p�A��A��.D�(�@���@�Q�Aq=�A�(�A��A��B���A�z�A  *D�����u��QP���ѾH�J��zd@�(A  PA   A3�'D33�33���(���©���	���5��(��)\?���@�H*D�pA��?�z��H�b�{>�H�:?���R��@ffA�3D��B�>B�QB=
�A{�A�z�A�BףB��+B��6Dף<A��GB�GmB�zLB)\�A33BffB
�JB��?B{47D���?
�WAR�NB{tB�GSB�zB  B33BףQB�:8D33�@���@R��A�_Bq=�B�cBH�Bff&B��/Bf6:D���@�z@A�[A�(�A
�~B��B3��B��3B�FB�X8DR���ף�>q=�@ף�@�z�A  aB.�B\�eB��B\�:D��%A��8@�-AR�nA���A�G�A33�BH�B�z�B{4;Dq=�?H�6A{~@ff>A  �A���A���A)\�B=
�B��<D��@q=�@)\�A)\'A��A��A��A��B
W�B�*>D��@��=AH�NAq=�A�}A  �A���Aff�A)\%Bff>D{n?�G�@�zLA��]A��A  �A�p�Aq=�A
��AH�<D\����̼�q=
�ff�@���@=
�AR�A�̎A���A��=D)\_@��U�q=���<@=
A�Q(A���A\�VAR��A P?D��@�'A��i@\��@=
A�p�A{�AH��A33�A�!=D���pݿ�G�?q=���z���(�?R��@ףA��A{9D����p�����33k�q=���̤�
�s�  ��p���;D��%A����Gm�H���33��H�2�  $��z�����?�~:D{����@��(��(���zD�ף���y�=
k��G�� >D�z`A�pA��AR�^@���  �?��@�˿��(�)�=D)\��)\_A�QA���Aq=Z@���=
�?�p�@�zԿ @BD�z�A��A�(�Aף�AR�B�£A  <A��A
ױA�,BD����{�A��A���Aq=�A�B)\�A337A��ARh@Dq=��
���=
A��A33�A��A�z�A��QA�(�@�P@D�p��{������A  Aq=�AR��A��A�KA�@D�p-@��@)\�������z@A)\?A���Aff�A33�Af�>D������������]�ffb�q=:@��5@���AH�FA��CD��A�=A�zhA\�bAH��@�G�@�̾Aq=�Aq=BfvBD{��  pA�(�@�p	A�A33�?��Y?�G�AR��A�@Dff��py�33�@�Gq����=
��  �����(Aq�BD�(0A{�?\���H�A��@33A�GAף @�p@\?AD=
���G�@�����,�q="A\��?R�n@=
W@�pm���AD��	@q=��{�@�p-�ff
�ףDAH�J@�(�@�Q�@3�>D�pI�=
'��G���̴���t��������R��R����>Dף ���q�33O�)\��\���z��  ��
�3�H�>��;D���{B�����\����Q��q=���p���z�H�F���>D��AA   @
�#���I�33'�)\�������t�  ���>D���R�A��L=�p���p�ffN��������{���%BD  �A��`A)\�Aff�Aף`A���?fff@�'�q=Au@D�Q��
�A�G�@�G�AףA���@���q=J�{�f6=D�O����)\O�{�����@�(L�ff���Q�����33>D��|@�z�ף|���5?R�޿{&A\�B?  ࿤pe�R�?D\��@�z$A��,�)\��G�@H�@��AH��@\��@� @Dף�?R��@\�:A�����G�R�A=
�@R��A�A�u?DH�*�����G�@
�A)\�  ,�  �@33S@R�vA\�CD33�A�[A��qA��A��A\�FA�̴@33�A���Aq}FD�;A���A���Aף�AףB�pB=
�A���A�zB3�EDq=j��� A��A�Q�A)\�A  �A��B�£A)\[A��BDףD�33�)\��R�BA  A{.A��A�G�AH�A)�=D\���H���{
�ff��������{��{����%@�a>D��@
׃��(��R��������)\��33��H�:?�=D����Gὤp�������
®G���Q��H������ @9Dף��q=�����=
�33E�
�S���$�R���{��=�6D�p!�)\������q=��ff<�\�m�33|��QM�R��{D7Dq=
@����{�����������3�d�\�s��D��U:D�QDAH�fAH�@
�_������a���
�3��zB�{�3D�(��  X��p5��p��=
�
�'��z�i��u����6D�=A33_�=
׿���>���������ff���z:�\�1D33���G�ff�ף��)\��{��)\C��(L���C��3D�p	A��D�H���{��
�_��G=�)\��  !���)�ף3D���A�GM��G��q=���(h���E����{#��5D��AH��@�p�A=
��33�@)\����������Ge�q]:D�Q�A33�A=
�AH�B�aA��A���=q=FA��hA�g7D�p=�ff�@��pAףhA=
�AR�@��`A�;���?�r7D{.>R�:�
��@�sA)\kAff�A��@�cA��8�e3D����Q��=
���p�H�z�  ��H��@��\�R���3;D���A�(hAH�jA��5@=
�A���A���Aq=B�G�AfV>D��PA{/B�z�A
��Aq=~AR�
B�(+B{)B�pKBH�@DR�A�³A��TB��B��B�z�Aff0B
�PB��NB=:CDq="A�z�A�pB�Q}B�z<B�(=B��B��XBffyBRX;Dq=������?�q=�?ff�A�pyA�(|AH�z@��A�6D�̚��K���"�\����(��33GAףp���e�H�v���/D�(���z6��L����)\f��(2¸]�q=��H���=
0D���>����H�4� ���R8��d�\�0�R�V�=
��HA/D��H�)\/�{���pA®ǟ� ����Qq¸=��z��f�0D\��@�(,@��E@�p���*¸����  Z���%�
�0DH���H�@��@ff&@)\��{,���р���[���0D�?
�#=
׻@R�.@�QH@�����)�=
�����h3D  ,A�z4Aף,A���A�WA{^Aq=F�����{}��6D�GeAף�AH��A���A���A�z�A��A�Q�?�G���`3D)\g�����)Aff2A\�*A��A��UA  \A�QH�
�4D33�@���=
�@�sA  |A�(tAR��A���A�̒A3C6D{�@ף8AH�:�\�6A�G�A��A���Aq=�A��A�R3D�(<�q=���Ga�H�j�33��ff&AH�.A=
'A�(�A�94D=
g@ff�Hᚿ��X@�1�ףP@�(`AףhA��`A�G6D�A�G=A)\�=�Q�@��9Aff6��7A
ױA{�A
75D�Q���p}@q=�@{��  0@33�@���=
�@�AN3D�z��ff>��k�)\���G=��z���z��  l�=
׾�9D���A�z�A��PA�(�A=
�A��QA�z�A�G�A33#A��8D�(,�{�A��iA��%Aף�A��AH�&A���A�¯A��6D�z���G)���eA)\�@{@=
+A��dA\�"@�A�6D)\���(�33?�  PA��@�̌?�AH�NA�?332D)\���Q���p�������p����@�\����������a2DH�:?����z����������(l��G5��py�
���=Z5D{>A��IA��������a�{��=
A��?�pm��
7D�Q�@��A���A���?�z?����   �33oA���@�7:D33KA��AR��A�G BffjA�zTA�z�@��,@33�A\�;D
׫@\��Aף�A
�B��B�(�A33�A�(,A�A{�;D���>ff�@33�A�G�A�(B{B�̢A
חA�p1A
W=D�G�@
��@
�GA��A���A�Q/Bq=2B��A�(�A��=D)\�?��@
�A��YA�z�A�GB��3BR�6B{�A=�<D�zD�����H�@�p�@ף(A��A  �A�'B�p*B=�>D  �@�k@���@�p9AR�>A�Q�A���A  B�BB3>D=
��z�@���?�(,@�A��AH�rA=
�A\�
B�c=D)\�33����	@�k���L>��@q=�@=
KA��Aõ<D{.�R���q=�������h��G!�ף�@33�@�A�Q<D�ǿ����ף��{��Q�ff��\����p-@\�B@�x?D��IA��0A�GAH�@R�n@�'A��@�zA�uA�@D��@��pA  XA�z,AףA���@H�NA��A�/A��>D{��)\?���A�� AH�@ff6@�p=?��@�p�@�j@Dף�@q=�?��q@��A�GmA��AA��A�Q�@�(dA�G?D����{@{N��E��p=A�z$A���@q=�@�p=@{�=D������1�\����Q�q=���G�@R�n@�G�?�p��{@D  A��<@��̿�p�@q=���@ףlA�SA�((A�>Dff��ff�?  �������1���ף��H��@���@�=Dff&���	�fff?�̴�33#��[�����p��{�@�b=D=
���%��z(�
׃�\���{B������,����{?D���@33�@ff�@  p�  �@��L�33���µ?ף��=�@DH��@��eA=
GAף<A��u@�pUA�G�@)\@�(A��ADR�^@�1A�̎AR�~A�QtAq=�@\��A�Q$A=
�@  CD��@�pAH�zA��Aq=�A=
�AH�>A�p�A{nA��DD���@ףDA�Q|AH�A��A��A�z�AH�AH��A BDH�6�  p��(\?H�@H�>A��Aq=�A=
�AH�A�ADq=���(L�\���ף�ף@@��)A=
�A33wA��lA  CD\��@  p@����    ��@�pAH�zA��Aq=�AHq8D��(���ףG�(��z�\��ff���(��q�9D{�@�(�
���(�H�5��(�R����H���W5DR���ffF��Z�33F��K�q=y��Z�{H��(:�q]4Dq=z�  ���z���(j�
�U��([¤p���(j�R�W��C0D33���z�����
��H��R���Ha��q=��H��R�0D�z4@�GY�����Q	¸��q=��{��q�����=�2D���@��A����)\/�ff��H��.��=
r�)\w�)1D=
��)\?�QX@�QP��p��{�ף��������®�1D�;@\�B�)\_@���@�p!�  `�R���33���B��q�2D��U@ף�@���>\��@ffA  ��\�*�  ���z���K5D�'A��\A��A�Q,Aff�A���A{n@\�B�q=��RH6D��|@R�fA{�A��A�kA  �A\��A�p�@ףp@R�8D  $A33cA)\�A{�A��A���A  �A�G	B)\�A3�6D�G��µ?
׫@�p}A�p�AH�A��A)\�A���A{6DR����4������8@��UA���A=
�A\�ZA��A55D)\O�=
����h�����33����!A)\WA��AR�&A3�6D=
�@R�@  ���G��?
ף@�pyA�p�AH�A)4D��!��z���(����%����=
����)\�@�A�q4D��?�Q�33C��G���Q��̌�33����Y�q=�@��2D=
���(��
�o��Q��(L�
�s�\����]����qm4D���@)\��\��?�p	��G�����p�)\���p�� P3DR����Q�?�����(<���P�\����-���T�=
���A7D�z|A�5A�A  4A�pMAR�.@33AR��@R�@f�8Dq=�@�̦A��A�Q�A\��A�G�A���@�QTA�z A�;8D����y@�p�A�sA���AffrA��A�(�@��AAf�:DR�*A  A�iA���A��A�(B\��A�G�Aff�Af6;D  �?R�>A  ,A�}A���A��A�(B\��A�G�Aq;D�Ǿ�(\?�z8A��%AH�vA��A  �A��B�p�AH:D{��\����U�H��@�p�@
�3A�(�A�z�A��A
�7D\����I�
�O�
�;�����)\��5@H�AffbA׃6D����)\c�33���Q���Q��
���ף �{>���LA��6D
׃?ף��H�R�����{��{��H���Q���Q����9D�p=A��MAq=�@��������p���p��  �@\��@H�9D��ѾH�6A)\GA��@  �{��\���\���H�@)L7D�G�
��ff@�QH@H�*��G1��Qt�\�z�\�f�R�6D���q=*���0�q=J?���?R�~�q=F�ף���� P7D)\�?��u=�Q�H��q=
@�(L@=
'��Q0�)\s�)7D���R��>  ���G%�
�+��̌?�Q@H�j��GA� 5D{��  ��(��=
��(���p��H��������9� 0D  �����  ��=
�����{�R��R����z��R�0Dף�?�����z������  ���z��\��33����R81D  0@�(�@��u��z������  ���z��\��33��1D��?H�@=
�@�z\��­�q=���G���µ�33 ®�2D�̌@��@
�A��)A{�\���=
��{��\����L+D)\���(���p���p��ff��33���;�33@�R�:��H*D������ף����������H���p,�33L¤pP¤�,D��!A���@H�����������m�
�O����R�#�)�-D�{@��`A
�A�p���zt�=
[�=
/�����z����/D�pA�QDA��Aף�A�p5�{��33��ff&��Q8� �1D=
�@��pA��AH��Aff�A
ד��Ga�)\�?
׃@�80D���{�?33A{ZA  �A��A��\���)\�>1D\��@
ף�{�@�z\A��Aף�A�(�A�̼�  ����0DH�ڿ�@)\?�R�n@�AA  �A���A�z�A����q3D�Q(A��Aq=NA���@  dAR��A�(�A\�B�QB��5DR�A��A
בA�z�A���A)\�A=
 B��Bq=8B� 7D��@�kA  �A�Q�A���A{�A
��A�GB  %B�^6D��A���@33;A�±A{�AR��A
יA���A�(	B3�7Dq=�@\�R@��	A�(�A�Q�Aף�A�G�Aff�A{BR�7D{.����@�G@H�A�̎A���A�G�A���A=
�A�L<D\��A33�A�½A��A  �A�B��7B��0Bq=AB=Z<D=
W>q=�AH�A�p�A33�A��A�B��8B��1BE=DH�j@�Qx@���Aq=�A���A\��A=
�A33B�GGB=�<DH�*�  �?H�?q=�AH�A�p�A33�A��A�B��=D
כ@��@
׻@\��@33�A
׻Aff�A�(�A�Q BHa=D�G�=
G@�G�>��@q=�@��A�­A�Q�A{�A�i=D�>ףп)\O@��?��@ff�@�(�A�̮A)\�A~>Dq=�@ff�@�(,@���@�z�@��A�QAR��A)\�A�u?D�w@=
A�A���@H�6A�(AH�FAq=JA��A��ED  �A���A��B�GBq= BR�B=
	BR�B\�B)<HD��AffBH�B�(-B�-Bף&B�:B�p/B�>B�HD��Q��zA�	B��BH�)Bff*B)\#B
�6B�(,B�GD�����G�\��@�QB��B{%B��%B\�B=
2B PGD��տ�7��(l���@�G�A�BffB��BH�B��IDH�A�( A���@��@�pyA)\$B
�3B�EBףEB�NID��ٿ)\�@���@��@�G�@q=^A\�B=
-B�Q>B��ID  @��?�#A��A��@�G�@��A\�&B=
6B\JD�G�?ףP@��?
�3A�A��A���@33�A��*B�JD�z���Q8?{>@\��?33/A�zA�GA�Q�@H�A3�ID33���Q���(ܾ���?���=��A{A���@��@q}ID���)\���!�\�¿H�:?�Qx�)\A�G�@H�@)LHDף������Q�������G���G������(|@�G@�ZID)\�@q=
�ףп��1��zD�
��\�B>�G��R�A �HDH�*���?�pM�����ff�����)\��R������GDffF�ף���ſ������������p�\���\����<GD��,�����������(���!�  4�ף8��z(��%GD�Q��
�C�����G�33�����)\'���9�ff>�ffFD)\?�ffV�����ff��=�H�����E�33W���i�3�ED33��G���̴����337���a�q=�\�j�  |�NFD���?\�¾�W�R�n������z�33C�=
��
�K�{�FDff&@ף�@{@�E�ף���u��������
׫���ED��y�ff��R�?=
׿�p������)\���G-�  X�fvED�z�=
���W�����  p����33�����ffN���EDff�?���p��������=���{������  ��7GD=
�@ף�@ff�@ff�?��i@q=�@�GQ@)\�>
ף�HJDff2A���A)\�A�̀A33CA��lA�AR�fAH�6A\�ID������1AR��A��A\��AR�BA�QlA��Aq=fAõHD�̤��¥�=
�@=
7A
�OAR�.Aף�@��Aף8A�ED��e�{���Q��q=�H�:�)\���(\�H������{�GD�p%Aף��R��33����?�p�@�A���@  @@.ED��!���u>��a��(��ff��ff��+�ף����L�q�EDR��?����p�?{F�q=���z�����ףp��(?�NHD�Q,A�(HAq=@  LA{ο�Q���G����@�GA~GDףP��Q�@  A��Y�
�A
כ��Q ��� ��̌?��FD�%�H���@�p�@�[���@ff����I�{J���GD�z�@��?�����A��0A��u?��4A
�C�)\��AID  �@q=A���@33s@�iA�z�AR��@ff�A�(@��KDR�.A)\�A�z�A
׏A�kA���A
��A=
�A���Aq�KD�p}�H�AH�rA\��A��A�[A  �A���A��A��HD��D���T��Q��7@�Q�@�@�µ?=
CAH�^A��JD�GA)\��=
��ff�@33/A�puA�(LA  A�(�A׃JD�+���@�̜��z�����@�z$AR�jA�pAA�GA��HD�p���p�
�#��E���T����=
7@  �@�p�@�ID�µ?  ���p���z�?ff.�q=>��Qx����@R�
ARHD�k�ף�H����%��G��Gi��y��̔���?��HDq=@\�����>ף��{��)\>R�B�\�R�)\��NHD�ǿ��Y?�5��z���G�  ��µ��[��k��HD���?�z�>��,@H�z��G�>)\�������(�>{>�q-HD���������>ffV�=
������Q ��Q���KD)\WA�p1A=
OA{6Aף\A��!A�z8A��@�(\@{dJD33����A��@�pA���@=
A�Q�@���@H���H�JD��@�����0A=
Aף(A�Aq=6AR��@{AID{������p!��W@  �?ff6@��?��l@
ף<f&ID�?�p��=
������x@\��?�W@��?=
�@õID)\@ף0@��R�.�\����(�@ףp@��@���@\_MDffjA��A�G�A�AR�>Aq=�@q=�A�G�A{�A��ND33�@  �A��A{�A�uA�(�AR�NA=
�A{�ARhMDR���)\>ףlAq=�Aff�A��A��@AR��@)\�A�ND�@��A��z$@�A��A
ןAףDA
�gAq=A\�PDH�:A��aAff
A  dA33�A��A�G�A�¿A)\�A3#OD{����@�p�@H�?���@��A���A���Aq=�Af�MDff��q=Z�H���ף�>����(?�(tA  �A�(�AfVLD  ��333����R�������
���z���((A  LA��LD�p@\�"�
���p��  ���z�����=
���KA�yJD\��ff��33C�33��R���\�b��;�����p9��^ID�p���Ga���=�����\���{��ף��33��H���ED�z��
׫�����p��R��� ®G=�\��
��)�CD)\���Q������z�ף�ף¤p9�33V��z'�fFCD���q=��=
��ff��
��  �  $���=�\�Z��HD�p�AR��A��YA��\�
���q=���u�\����(����JD33A=
�A�Q�A�z�A  �@�@\���
׫���!��eID�z��
�c@���A33�A)\�A�G�=����_��(<��CDff��������)\O�33�{����H���{� �@Dף�)\� �q=�����=
'�)\������N<Dq=��\����zQ�=
k�q=C�=
���������=
Q�=�AD��Aq=Z@�(���p���G�����{�������(T���BDq=�@{�A�AR���H���   �ff��)\���p�=�DD���@  0A��B\�fA
׻@�p��\�����}���@�AD=
#��Q��)\O?  �A=
�@q=������=
��z��33DD�GA{��q=�@q=Aף�A��PA�Q�@�Q���p�� @DD��L>�zA�z��ף�@�pAq=�A  TAR��@R��� �BD  ��������A@���H�z���u@q=�A  �@����CD�p�?ף��q=���Q�@�½�  �>q=�@{�A�A�rDD��@�G�@q=J?�p}?�A�p����@{*A�GB)\ED��i@)\A=
'A{�@�z�@�WA��Q@)\A�zdAR�AD��X�\��33���G����R���Q���z$�33��?D��8�H������ff~�R�j�)\���£�q=:�ף��q?D�>R�6�
���ף���Q|�ףh��Q��R����(8��BD�(\Aq=^A��@�3�q=��ף ��G�����\����BD\���;A��=A���>�(T���������z4����EDףtA  TA{�A��A�pyA��@�µ@
�3A�GA��FD��l@��A���A��AR��A�Q�A)\�@{A=
oA�\JD�zdA
׏A{B
��A��3B�z4B�GB{�A�G�A��IDffƿ�KA�p�A���A�p�A��-B�G.B{ B��A��KD{�@�z�@)\�A���AףB�zB�KB=
LB
�B�xLD33#@
�A=
A�µA)\�A
�&B�BR�UBq=VB��KD)\�R��>  �@ff�@
ףA�p�AH�BR�B��LB�uJD����� �  ���Q�?��>R�jA���AףB���A  KDq=*@��I��z����5�33�@33C@ף�Aq=�A�GB��KD��,@��@ff��(,�)\����@  �@q=�A
׽A�KD  @�����=
@��\�{����H����@  0@q=�A�kJD�G!�ף���z4�
�#�=
��)\����
��?�k>3�IDף��̌�����ff��\��33���p!��G����̽�{JD�Q@  �>�G�ף���z$��Q�==
��R������{$ID���R�N�����(��{*�����ף���G1��U�R�HD�Qؿ�����p���������E����R����QL��FDH�&���A�
׋���u�
׉�  ��  ��ff������ JD��xA�(�@�(\@����=
W>��տ{��{�����H�ID��u��uA�z�@��L@q=
������z��������HAKD  �@�Q�@\��Aq="A33A��E@=
�@��U@��Q?qMND=
CA��A���A=
Bף�A��A�ztA�G�A�zxA��ND�(�?\�bA�G�A)\�A��
Bff�AH�A  �A=
�A��ND����L?
�OA��A  �Aq=B=
�A��Aף�A
wODffv@�+@�̔@R��AR��A�̮AףB
��A�Q�A�aOD�����Ga@ff@q=�@{�A{�A�(�A�QB33�A)�OD�z�?q=�?�@ףp@)\�@)\�A)\�A�p�A��Bq�NDR�^��z�������?)\�>  @=
gA��A���AHAOD��?����\��=
W�ף@@��?
�s@  �A  �A�ND�̔���E�q=����������ѿ
�3�=
W���5A�
OD33s@��Y���5?�G1�{���Qؿq=
@�p}?�p=@�SD{�A�z�A\�vA�A
�WAffnA�iA)\�A  �A�aTD33�@H�A�G�A{�A\��AR��A  �A)\�A�(�A͜VDR�A�Q`Aq=�A�QB�p�A���A{�A)\�AR��A�>VD�(��ff�@��HA�z�A�pB��A�(�A�Q�A���A�ASD33?�R�V�  ����?H�A�G�A{�A\��A�paA�MD��������z
����̤�H��q=
�{���Qh�)MD��h�R����(�=
�R�������)\�����G�\�KDff���p	��Q����&�
�,��(	����H�N�{���LD�GA@=
��q=���(��H�� ��(��)\��\��RMD)\�?�z�@��u���l�33��ff®G�33��ff���gMDR��?=
W@�(�@=
�?�p��G���p��Q®G���NOD��@��A�/A
�_AףA�̬@��|�  ������RxODff&?�(A  A��9Aq=jA=
A���@ffr�����׃OD�Q8>�zT?=
AH�A��<A�mA��A)\�@�o�f�PD�G�@=
�@
׳@�SA�kAR��AH�A\�jA�Q0Aq�QD��@ffA�GA�AR��Aף�A���A�½A�(�A�RD���?�¥@�Aff"A��,A�G�A33�A�(�A�Q�A{TQD�p-���ѿ{@�Q�@{�@�pA33{A��A�z�A�RD33�@��@�pM@q=�@��AAףDA=
OAff�A�Q�A3�SD�(�@�Aף�@H��@33;A��A)\�A\��A�p�AHQTD{.@33�@33?A
�A��$AR�fA��A��A�Q�A
�TD��@��@=
Aף`A�G5AffFA{�Aff�A
׫A �TD�Q���p�?ff�@�GAH�ZA�/Aף@A33�A��A��SD�(L�33c�Hế�G�?�z�@
�'A���@��A)\OA�jTD{�?q=���Qؿ���>�G@  �@��EAq=A)\+A�;SD����Q8�q=������H��)\Ͽף0@��@�̜@ 0VD�=A\��@=
A  �@�z�@)\�@33#A�GiA�p�AfvTD�����p�@�Q8>\�@33���G���z?33S@���@ETD�E��p���̄@=
�\��?�������\�B���!@H�SD�翸%�����@���q=
�R�n�H��   ���TD�Qx@�z@ff�?33��=
�@�p�?��U@��>=
W��hUD)\@
��@��@\�r@=
G�)\A{~@\��@��(@�uVDff�@{�@�)A�(A��@��?\�NAR�A�z A�YWD�(d@�z�@{$A�(bA33EAH�8A���@�̃A��;A%XD33K@��@=
/AH�VA�z�A  xA�kA\��@33�A=zXDq=�?�(�@�A�QDA�(lA��Aף�A�z�A\�A�YDR�@
�s@��@��(A  lA��A���A�z�A�Q�A �YD{�?H�@�p�@�	A\�BAH�A�̖A
׵A)\�A�ZD�@R�~@R��@�G�@�p/A�zhA
וA�©A���A=�XDR����������   @�U@�(�@�!A�QdA{�A��TD�(��
ױ�H��  ���Q|�=
g�q=4�ff��  `�3sTD{��H��\�������R���H���zl��9�ף ���QDff"�
�'�{��H����������{���p��=
����PD�Q��\�j�  p�{�¤p�  ��(�������QD�W@�둿ף4�{:�33���p���=
��33��R(RD�@��@H�z?R���(�q=����������{���wTD
�A��5A�kA�#A)\�=q=���Q��  ��=
���^TD�Ǿ��A�/A�peA�GA
ף���(��p�����3�SD=
׿�z��p�@ףA\�JAffA   ����H�� 0VD33Aף�@�(�@���A��AH�A�̈Aff�@��@{$UD��ף�@��E@��,@=
?A��`A�p�AR�NA�G1@=:TDq=j�H���{�?�����u��zAff&A�Q\A�(AFSD�(t�33���z:��-��Q���̘�H�@R��@�GAf�RD�?������'�ffj�ff���(��ף���(�?��e@�lRD��̾�GY�R�����-���p�33������R�����?R�QD���{�H���z�=
K������p�����
��q]RDq=�?��u�
�#�ףh�ff����1�ףt�H����Q ��RD
ף����>\�¿�����G����q=F�\���
���R�UD33oAR�ZA  tAH�VA�zPA\� A=
�@
�#@)\Ͽ\UD�둿��\A�zHA��aAףDAq=>A�QA\��@�µ?� VD�G!@ף�?ף�A��pA=
�A��lA\�fAף6A33�@WD�zd@H��@ff�@33�A���A���A=
�A
׏A��oAͼVDף���(@R��@�zt@�(�A��A\��A  �A�̆A��[D
ןA�̖A)\�A��Aff�A  BH�B33B��B�5^D\�A��A{�A�QBffB
�BףBB�=B
�CB�H^D���>)\#A��A�z�A�B��B=
B
�CBR�>B)�^D33@ff&@�(HA��BH��AR�B��Bq=B=
MBH!gD�QB�BR�B)\6B�#�BH�B�B\�B�ǊB{4iD��A�%BR�.B��/B\�WBq��B�z�B���B���B�EjDף�@�IA��6B��?B  ABףhB�G�B�B�(�B�mD��5A{zA�p�A=
dBq=mB�pnB=
�B  �Bq��B�nDף�@=
�A33�A���A�}B�(�B�B{��B=��B�mD{��q=�?=
SA��A{�A)\kB\�tB��uB3��B)�oD\�A{~@
�#A�̬A���A�B ��B��B3��Bq-oD�p�����@R��?�(A���A��A�� B=��B�#�BõnD)\�ffv���@���=�z�@  �A�(�A\��A��}B��nDff�?��Q�333���@�?{�@ff�A\��A���A��nD  @����>��ȿ33c���@=
�>{�@ff�A\��A�oD\��@�U@�(�@ף @�?=
A  �@�Q,A=
�A
'qD�z�@�A�A�QA���@�p�@�GeAq=A�G�A�kD���ff���O��[�R�J�ףh��(�������H� �mD�GA��a����z���z��H��R���=
��Ga>��nD�̄@�GA)\�q=��H���R���\�B�
���G��{qDR�A�YA33�Aq=
�33�@H�AH�A�A��@3�rD)\�@ff~Aff�A=
�A{�@�G5A\�vA\�jA)\{A
gtD���@ףXA��AH��A��B  PA��A�³A�­A
�wD  dA�z�A�Q�A
�B�p%B��FB  �A\�BH�BHQxD�z�?\�zA�µA���A�zB{+BffLB�G�A33B�
yD��9@��@�z�A���Aff B{&B�6B  XB�z�A��vD�G�����ף���A�Q�A�(�A��B)\B�3B�wD���?�(��)\���zd�H�*A��A���A\�B�(B�yD\��@�zA��L�ff6@�Q�@{�A\��A33 BH�%BR�zD�Q�@�pyA�Q�AR��@��)A�Q@A�(�A�QBq=B�+yDff��)\?q=A�pA\�?q=Z@q=�@\��A=
�A�ezD��@\��=
�@��TA  hA�p�@�A�A
׿A�{D�(,@33�@���>\�A
�A��A��A�(0AR�FA��rDף����z��=
�  ��)\����ff��33���spD���H�)¸��z®G(�q=	�
���q=���p	� �rD=
A\��>��R����p���������Q��R�����qD�z$�
��@�(�ff�ף�  ��������H���pD  ���%�ףп=
�ff0�ף%�  ���.�®�oD�ǿ���{>��(L�  8�ף6�H�+�q=�=
5��sD�̄AףpA�G�@=
�@\�VA33�@�z���������\?rD)\����%A��A��?�G�����@ף���(�ff�åtD��A�W@�A�G�A\�*A�pAq=�A�A����'tD�(���(�@33�?  �A��A=
A
��@��lA  �@ÅvD�A  �@�̈A��-A���A�G�A�G�A�pyAq=�A�sD  `�����  ��ffF@�QH��WA\�>A��@��?�mD  ��  �q=��  ��33��=
���z(��pA�R����@pDR�NA�G1�ף����y�ף��)\��)\c���@�zT?{tjD����Q$�{	�{A�33�{#�)\�����ff���khDq=�ף���G��ף)�ףa�;�ףC�q=��(6�q�eD�#�H��33&�=
���RB��ףd��l¸F¤iD��LA�%@���  ���G}��Q��QW¤p1��Q9�)�gDq=���AR���{"��G�33����1�i�R�C��3kD��QA��A�̪A{2A)\?@��������q=���5� �mD=
#A�z�A��A�Q�A\��AH�RA�( �q=:@R����anD��!@�KAR��A�(�A�GB�̾A)\{A)\��{�@=�qD{bA�G�A���AH�B��B��@B��BR��A���@�-rD)\�?  sAq��A�B�A)$B{�B�EBf&B.�A=:tD{A  A=
�A�G�AffBH�DB��2B��eB��<Bq�sD33s����@��A�p�A��A��B{AB��.B  bBͼvD
�/Aף A)ܑA�Q�A�B��B\�8B=
mB��ZBH1xDq=�@�z�A��}A�k�AH��A��B{'B
�OB�(�B)xD�z���@
ׁA�ztA�ǻAq=�AףB��$B�MB�uxD33�?���?�z�@=
�A�p�A���A�p�Aq=!B)\+BדyD=
�@
��@�G�@��5A�̲A33�Aq��A33�A�3B�ArDq=���z���G�����)\�������(��  �>)\�?�SqD{n�  �q=��=
��������ff*���9�{Z�� nD��L��(��337��Q%�R��=
!�	���33���oDq=�@)\��33+����=
¤p�	�����=
�� `lD�GM��Q���z��q=��q=S�)\A�:�{=���%�ףlD��?�Q<�ff��  ���³�  O¸=��6�
�8���lDq=�?��@=
+�
כ�)\������J���8�332�3kDH����Q��ff��q=��)\C�{��
���=
h��(VrlD��@��쿸E��z�>ףH�=
���(�����{R� plD
�#�ff�@���)\O�  �>�GI��Q���z��q=��)lD�ǿ��̿��x@��\������q=b�������kD����p��R����Ga�����)\���p��  ��H�F���fD����̤��G���������ff���·��G����
�ÅhD  �@
����a�\�z�33{�)\#�ff����\�v���jD�pA�puA33���Q��q=�����)\��)\��(���jD��Y���@
�gA   �����p��R���{����HlD{�@H�@H�bA�p�A{�@
ף=�p��\�¿{~@=�gDH��R�&��Q4���q=�@R�F�q=��R���=
����gD�p��R���ff>�  L�q=z���%@ff^�{��\����7fD����G��33������z����{.����\����kDq=�A�cA
�KA�w��z@�(�?��$A�z�A
ף>�kD�@��A33�AR�nA�Qؿ  �@��i@
�GA��A�8jD{��ף`��(�A)\+A�A�(���Q��\�����@ �hD�z���GA�ff�{Aq=�@��5@�Q\�\���H��RhD���(�33g��QD��Q�@���?ף�>���33� �iD
��@  �@�둿\����̔�{nA�A�pA�Q�R�kD�(�@  xA{RA��@ff�?R�N@{�A���A�A\�nD��9A��AH��A���A�̒A\�JA�pmA�z	B�z�A\�mD  P���A��AH�A��A��qA\�A�p9A���A�^kD�((��(\���	�)\�@��UA�/AH�@�̌�q=�?H�kDq=
?���S�{οף�@q=^A�Q8A�(�@)\��xmD
��@\�Aff�33���Q�@q=bA{�A��A  PA=ZkD���(�)\���G)��G]�{���@�zTA\�.A��kDף�?33��\��?��?33�33?�{���G�@\�rA��lD=
W@��@)\O��(�@�p�@H���p	��GA@ff.A)�mD\�r@���@�zA��?R�A)\Affƿ�������@RkD��$�ף��q=J�
ף��(�����̬���=���q���iDH��ffr���5�   �
�����i�)\��{������ phD�G��{&����33��ףX�\�:�����QD��;�
7jD��@���?�GQ��GY�ף����������zP������lDH�*A�Q�A  DA��@��9�
�c?  �@�(�@ff�3�lD
�#<=
+Aff�A�(DA�p�@��8�fff?�Q�@�z�@��kD�G���������@�(XA=
�@�QH@������X������iD���QT��(T��%����@�G��33���G����E��[jD��I@\�����!���!���?���@��@��,��(P���oD�³A���A��A��EA��EA�Q�A33�AH��A�(�A�3qD��@=
�Aq=�Aff�A{�A�(�A���Aq=B�(�AnsD\�A�]A�(B��B��A)\�A�p�A�pBH�/B�qD  ��q=*@q=�@�Q�A��B��A)\�A�p�AH��AsD��@��ѿף�@H�BA��
B33B\��Aq=�A�Q�A�qD����ף0�{���(�?�(�@���A  B�(�A
יA)LvD�G�A��QA�A�7A=
�A�Q�A=
?BףKBR�'B�hxD33AH��A�z�A)\�A)\�Aף�A��B
�`B�pmB�xDq=��
��@q=�A
סAR��AR��A  �AףB�[B3�vD�Q��H��=
'@�(�A�{Aף�A�GaA��A33�A�i{D�̎A�pUA�(@A��A�zB�GBR�B�p�A)\#B�i{D    �̎A�pUA�(@A��A�zB�GBR�B�p�A��{D{@{@\��A��xA�cA�p�A)\$B�(B��!BfV|D�p�?��l@��l@ff�A�Q�A)\{A�G�A�G*B{B �|D33�?�QH@33�@33�@���A��AH�A�z�AH�0B\/}DR��?��X@
כ@H��@H��@��A�p�A�̘Aff�A�uzDff.�\���Q������
�s�
�s�ף`A�zA33A{D{DR�N@�p���½�����333��z��z��(�A�(LA�bvDq=��ff����������z��ף��H��H��ף�\wD��<@ף����U�  ��{��H��=
���G���G��q�wD{>@�p�@��Y�{&�q=���Q������G��=
c���wD
�#��;@�(�@ffZ�R�&�\���ף���p������q5�D  )B
�(BR�4B�@B���Aף�A�p�A)\�A\��A)$�Dq=
�
�&B�&B\�2B)\>B�z�A�Q�A��A=
�AH!�D�Q���G!��z&B�Q&B332B  >B���A���Aff�A3S�D��A�AH�AR�LB\�LB�pXBq=dB�B=
#B
_�D�p�>H�A�pA��A33NB=
NB��YBR�eB��B�?�Dף�@�z�@���AH�A\��A�GjB�jB  vBf�B�U�D�����z��
ף=q=A��A�(A=
MBH�LB��XB{��D)\A��@R�AףA�̚A{�A�AH�sBR�sB
��D�p���G�@�G�  �@
ף@H�jA�piA��`A33aB̈́�D�zd�
���(�?H��=
�?ff�?��1A�Q0A�'Af&�D��<�ף��=
3��p��ף�\��33��\�A�A{4�D�G�>ף �\���  ,�������q=����u���	A��D)\���G�������4���ף���GY�����{��
�D��@A\��@ף�@q=�@  @?H�z��G�@\�¿  �@{|�D�GI���  ������(��G=�  ���G����a�U�D�p����\�  �)\���G��
����P�
׍��Q ��I�Dq=zA\�fAq=�?q=^A\�
A��A���@�%@���q}�D)\�?{�Aq=�A��\@�(xA�z$A�+Aף�@ff�@��D33�@=
�@H�A=
�A�� AH�A{nA�uA��EAR �D
�wAR��A��AffB�zB�Q�AffB���A�z�AH�D��ף�@��5A
�OA=
�A33�A�mA=
�A33�A)��D�G���
¸��ףh�R�N�{.@R��?�p1��Qx?q��DHᚿ����)\��̢�  |�{b��G�?)\�>��D�="�D��1���D�H��;�������
����p��-���|D33S�ff��{���M�\�p�2�33 �R���Q��f�~D)\�@=
��\���q=����.�ףQ��®G���
�D��QA�̦A���@��\����\����(�ff������=��D��=A���A��B  �AffA{�@�����33?�),�D��@�G�Aq=�A=
B�z�A)\SA  @A\�^�=
��
�D��%@���@  �A�zBff B33�A��|A�piA�5���D��L��(�ff�@ff�A)\�A��B���A��IAq=6A{l�D�'@�z�ף @q=�@)\�A�Q�A{B\��A�sA
/�D\��@33A  �@�pAffNA  �A�zBff6B��B  �D�zhAH�A
׹Aq=�A���A�p�A�B��QB�pBᢇD�pQA���A��B�GB�zB
�B{"B�zQB���B���D������@�z�A��A{�A�z�A33�A
�Bq=<B�1�D=
���z8���?R��A)\�A�Q�AR��A�p�A���A3ۅD�p-�H��
�c�33��{VA��Aף�A=
�A�«A���DR���ff��R�.�
ׁ���H�q=6A�AR��A��A3;�Dף@�  ��R���H�^��������{A)\gAף�Af��D�̴�\�
�ff*���U�ף�������<�R�.@��Af��D  ��ff�\�N�ffn�H��ף�����ff��\�¿ͬ�D33#�����33G�)\w������G��=
������̔��&�D�z�A{�A�(PA��@33�@=
@33��q=��{>��p�D�z�A�zB�GB\��A)\�A�G�A)\�A��A
�;AR��D��@ff�A�p)Bq=Bq=B�G�A33�A�G�A���AHɊD�zhA�(�A�QB\�cB)\YB)\HB��1BR�%B��B�F�DH�z@���A��A  $Bq=sB=
iB=
XB�pABff5BH�D�µ�   @q=�A�(�A�QB\�mB)\cB)\RB��;B{ĊD��)�q=�����{fA���AR�B��bB��XB��GB)��DR��?��h�����?���AH�A�B��iBR�_B�T�D�(�A{�AH�A��AH�A\�B�B��eB ��BfV�D��L=\��A�z�A�G�A��A�G�A��BR�B��eB3c�DffA33A���A��A�z�A��A�z�A)\-B�Q>B
��D�QH�ף�@q=�@R��Aף�A�p�A{�A�p�A
� B ��D���@�̌@��LA��MA���A�pB��A�Q�A
�B��D�G�A��A�z�A
�B=
B�QB{XB�zMB��GBq�D�(@A�B�� B�GBH�5B{6B{��B\�B�}BHi�D�(���(�@�Q�A�GB�� B)\"B\�"BףmB��tB{ēD��-A=
�@
׏A\�B�8B�(,B��MB��MB��B�q�D�p�@�(�Aq=6A33�Aq=0B)\NB
�AB�pcBףcB\O�Dq=��H�@=
sA��$A\��A��+B=
JB�=B�_B�+�D���=
#�ף��\��@���?)\SA�zB��%B{B
ǓD33�@�Q��H��
ף=H�.A���@�z�AH�B  9BᲓD�G!�=
�@�z��=
�������$A�p�@�p�A)\Bf�D\�y�{|��h�\���R���{�)\P�H�c�
�3�R��D��I�������R8�¤p������\������Í�DR�zA{��Q��{���H��B�k��R����D�(����U��(�)\�¸�������f&��u��ff{Dq=G�����ף|�33��q�/ø�0��+�H�4���5��m�D�QB�?�  $�ף��=����(
�=�
ä�����уD{2A
�CB��Y�����33c��G��\���Q�¸���{ăD=
׾)\+A�(BB\����Q����i����f���{ �f�D�z�A��A�(�A33�B�Q�A�;��(|@  :®��k�D�G�AH�B33BR�?B��B��B���@�̴AR��� ��D�z�@ff�A�p%B��#B�GPB�̳B)\ B�1A���A��D�pA�CA��A��EB�DBףpB���BR�@B�G�A 8�D\�"A  �A��A33$B�pnB��lBף�B�L�B)\iBͼ�Dffv����@fffA�Q�A��B=
_B)\]B���B���B��D��A�peA  �A)\B��B\�]Bf�B\�B�Q�Bf��D��yA33B���A�p B��@B)\QB  �B��B�G�BR��D����G)���@ףp@R�^A{�A33�Aq=3B�z}Bq��D�pY���)\��)\���G����>R�A��HA���A�D=
�ff8�{���R�b����)\)�R� �R�������{��D�KAq=���z�q=n���/�{��H���������5�R؈D��)A�̺A\��  ����C��Q�q=r�����G�͜�D�z�@{�A���A�GA�H��33+����  ���M�Ha�Dq=bAq=�A33�A�.B��1A{��G����P��z�@),�D�peA
��A�z
B��4BH�gB��A��=A�e���?*
dtype0*
_output_shapes
:	�


z
MatMulMatMulMatMul/aVariable/read*
T0* 
_output_shapes
:
�
�*
transpose_a( *
transpose_b( 
N
addAddMatMulVariable_1/read*
T0* 
_output_shapes
:
�
�
<
ReluReluadd*
T0* 
_output_shapes
:
�
�
y
MatMul_1MatMulReluVariable_2/read*
transpose_b( *
T0*
_output_shapes
:	�
d*
transpose_a( 
Q
add_1AddMatMul_1Variable_3/read*
T0*
_output_shapes
:	�
d
?
Relu_1Reluadd_1*
T0*
_output_shapes
:	�
d
{
MatMul_2MatMulRelu_1Variable_4/read*
T0*
_output_shapes
:	�
2*
transpose_a( *
transpose_b( 
Q
add_2AddMatMul_2Variable_5/read*
T0*
_output_shapes
:	�
2
?
Relu_2Reluadd_2*
T0*
_output_shapes
:	�
2
{
MatMul_3MatMulRelu_2Variable_6/read*
_output_shapes
:	�
*
transpose_a( *
transpose_b( *
T0
Q
add_3AddMatMul_3Variable_7/read*
T0*
_output_shapes
:	�

R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
n
ArgMaxArgMaxadd_3ArgMax/dimension*

Tidx0*
T0*
output_type0	*
_output_shapes	
:�

J
softmax_tensorSoftmaxadd_3*
T0*
_output_shapes
:	�

̯
!softmax_cross_entropy_loss/Cast/xConst*�
value�B�	�
"Ю              �?      �?              �?                      �?      �?                      �?      �?                      �?              �?      �?                      �?      �?              �?                      �?              �?      �?              �?              �?                      �?              �?              �?              �?              �?              �?              �?              �?      �?                      �?              �?      �?                      �?      �?                      �?              �?      �?              �?              �?                      �?              �?      �?              �?                      �?              �?      �?              �?              �?                      �?      �?                      �?      �?                      �?              �?              �?      �?              �?              �?              �?                      �?              �?      �?              �?              �?              �?                      �?              �?      �?                      �?      �?                      �?              �?              �?              �?      �?              �?              �?                      �?      �?                      �?              �?              �?              �?      �?                      �?              �?      �?                      �?      �?              �?              �?              �?              �?                      �?              �?              �?              �?      �?              �?                      �?              �?      �?              �?                      �?      �?              �?              �?              �?                      �?              �?      �?              �?                      �?              �?              �?              �?      �?                      �?              �?              �?              �?      �?              �?                      �?              �?              �?              �?      �?                      �?              �?      �?              �?                      �?              �?              �?              �?              �?      �?                      �?      �?                      �?      �?              �?              �?              �?              �?                      �?              �?      �?              �?              �?                      �?      �?              �?                      �?              �?      �?              �?              �?              �?                      �?              �?              �?      �?              �?                      �?      �?              �?                      �?              �?              �?              �?      �?              �?                      �?              �?      �?                      �?              �?              �?      �?                      �?      �?                      �?      �?                      �?              �?      �?                      �?      �?              �?              �?              �?              �?                      �?      �?                      �?              �?      �?              �?                      �?              �?              �?              �?      �?              �?                      �?      �?                      �?              �?              �?      �?              �?              �?              �?                      �?              �?              �?              �?      �?                      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?                      �?      �?              �?              �?              �?              �?                      �?      �?                      �?              �?      �?              �?              �?                      �?      �?              �?              �?              �?              �?              �?                      �?      �?                      �?              �?      �?              �?              �?              �?                      �?      �?                      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?                      �?      �?                      �?      �?                      �?      �?                      �?      �?              �?                      �?              �?      �?              �?              �?              �?                      �?              �?              �?              �?              �?              �?      �?                      �?      �?                      �?              �?      �?              �?              �?                      �?      �?                      �?      �?              �?              �?                      �?      �?                      �?      �?                      �?              �?              �?      �?              �?              �?              �?                      �?      �?                      �?              �?              �?      �?              �?              �?                      �?      �?                      �?              �?              �?      �?              �?              �?                      �?      �?                      �?              �?              �?              �?      �?                      �?              �?      �?                      �?      �?              �?                      �?              �?              �?      �?              �?              �?                      �?      �?                      �?      �?                      �?              �?      �?              �?              �?                      �?      �?                      �?              �?      �?                      �?              �?              �?              �?      �?                      �?      �?                      �?              �?              �?      �?                      �?      �?                      �?              �?              �?      �?              �?              �?                      �?              �?              �?      �?              �?              �?                      �?      �?              �?                      �?              �?      �?              �?              �?              �?                      �?      �?              �?                      �?      �?                      �?              �?      �?              �?              �?                      �?              �?      �?              �?                      �?              �?              �?      �?              �?                      �?      �?              �?              �?              �?                      �?      �?                      �?      �?                      �?      �?              �?                      �?              �?              �?              �?      �?              �?              �?                      �?              �?              �?              �?              �?              �?      �?                      �?              �?      �?              �?              �?                      �?              �?              �?      �?              �?              �?                      �?      �?              �?                      �?              �?              �?              �?      �?                      �?      �?                      �?      �?                      �?      �?                      �?      �?              �?                      �?      �?                      �?              �?      �?              �?              �?                      �?      �?                      �?              �?      �?              �?                      �?      �?                      �?      �?              �?                      �?      �?                      �?      �?                      �?      �?              �?                      �?              �?      �?                      �?              �?      �?              �?                      �?      �?              �?              �?              �?                      �?              �?              �?              �?              �?              �?      �?                      �?      �?              �?              �?                      �?              �?              �?              �?      �?              �?                      �?      �?                      �?      �?                      �?      �?                      �?              �?              �?      �?                      �?      �?              �?              �?                      �?      �?              �?              �?                      �?              �?              �?              �?      �?                      �?              �?      �?                      �?              �?      �?                      �?      �?              �?                      �?              �?              �?      �?              �?              �?              �?              �?                      �?      �?                      �?              �?              �?      �?                      �?      �?              �?              �?                      �?      �?                      �?              �?      �?                      �?      �?              �?              �?                      �?              �?              �?              �?              �?      �?                      �?      �?              �?              �?              �?              �?              �?              �?                      �?      �?              �?                      �?              �?      �?                      �?      �?                      �?              �?              �?      �?              �?              �?                      �?              �?      �?                      �?      �?                      �?              �?      �?                      �?      �?                      �?              �?      �?              �?                      �?      �?              �?                      �?              �?              �?              �?              �?              �?      �?                      �?      �?                      �?      �?              �?                      �?      �?                      �?      �?                      �?      �?                      �?              �?              �?      �?              �?              �?                      �?      �?              �?              �?                      �?              �?              �?      �?                      �?              �?      �?                      �?              �?      �?                      �?      �?                      �?              �?              �?      �?              �?              �?              �?                      �?              �?      �?              �?                      �?              �?      �?              �?                      �?              �?      �?                      �?              �?              �?              �?      �?              �?                      �?      �?              �?                      �?              �?              �?      �?              �?              �?              �?                      �?      �?              �?              �?                      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?                      �?              �?      �?                      �?      �?                      �?              �?              �?              �?      �?                      �?              �?      �?                      �?      �?              �?              �?                      �?              �?              �?              �?      �?                      �?      �?                      �?      �?              �?              �?                      �?              �?      �?                      �?              �?      �?                      �?      �?              �?              �?              �?                      �?              �?      �?              �?              �?                      �?      �?              �?                      �?      �?                      �?              �?      �?                      �?              �?              �?      �?                      �?      �?                      �?              �?      �?              �?              �?                      �?              �?              �?              �?      �?              �?              �?                      �?              �?              �?      �?                      �?              �?              �?      �?              �?                      �?              �?              �?              �?              �?              �?      �?                      �?              �?              �?              �?              �?      �?                      �?              �?      �?              �?                      �?      �?                      �?      �?                      �?      �?              �?              �?                      �?      �?                      �?      �?              �?                      �?      �?                      �?      �?              �?              �?                      �?      �?                      �?      �?              �?                      �?              �?              �?      �?                      �?              �?      �?              �?              �?                      �?      �?              �?              �?                      �?              �?      �?                      �?      �?                      �?      �?                      �?              �?      �?                      �?      �?                      �?              �?              �?              �?      �?              �?              �?                      �?      �?                      �?      �?                      �?              �?              �?      �?                      �?              �?      �?                      �?              �?      �?              �?                      �?      �?              �?              �?              �?                      �?              �?              �?              �?              �?              �?              �?              �?      �?                      �?      �?              �?              �?              �?                      �?              �?      �?                      �?      �?              �?                      �?      �?              �?              �?                      �?              �?              �?              �?              �?      �?              �?                      �?      �?                      �?      �?              �?              �?                      �?              �?      �?                      �?              �?              �?              �?              �?      �?              �?              �?                      �?      �?                      �?      �?                      �?      �?                      �?              �?      �?                      �?              �?      �?              �?              �?              �?                      �?              �?      �?              �?              �?                      �?      �?              �?              �?                      �?              �?              �?              �?      �?              �?                      �?              �?              �?              �?      �?                      �?      �?                      �?              �?              �?      �?                      �?      �?                      �?              �?              �?      �?                      �?      �?                      �?              �?              �?              �?              �?      �?              �?              �?                      �?      �?                      �?              �?      �?              �?              �?              �?                      �?      �?              �?              �?              �?              �?              �?                      �?              �?      �?              �?                      �?              �?              �?      �?              �?              �?                      �?      �?                      �?              �?      �?              �?                      �?              �?              �?      �?              �?                      �?      �?              �?                      �?      �?                      �?      �?                      �?      �?                      �?      �?                      �?      �?                      �?              �?              �?              �?      �?                      �?              �?      �?              �?              �?                      �?      �?              �?              �?              �?              �?                      �?              �?      �?              �?              �?              �?                      �?              �?              �?      �?                      �?              �?      �?                      �?              �?              �?      �?              �?                      �?              �?      �?                      �?              �?              �?      �?                      �?              �?      �?              �?                      �?              �?      �?              �?              �?                      �?      �?              �?              �?                      �?      �?                      �?              �?              �?      �?                      �?      �?                      �?      �?                      �?      �?                      �?              �?              �?              �?      �?              �?              �?              �?              �?                      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?                      �?              �?              �?              �?      �?              �?                      �?      �?                      �?      �?              �?              �?                      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?                      �?              �?              �?      �?              �?                      �?      �?              �?              �?              �?              �?              �?                      �?      �?                      �?      �?                      �?              �?      �?                      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?                      �?      �?              �?                      �?      �?                      �?              �?      �?                      �?              �?              �?              �?              �?              �?              �?              �?      �?                      �?              �?              �?      �?                      �?              �?      �?              �?                      �?      �?              �?              �?                      �?      �?                      �?      �?                      �?      �?              �?                      �?      �?              �?              �?                      �?      �?                      �?              �?              �?              �?              �?              �?      �?                      �?              �?      �?                      �?              �?      �?              �?              �?                      �?      �?                      �?              �?      �?                      �?      �?              �?              �?              �?                      �?              �?      �?                      �?      �?              �?              �?                      �?              �?      �?              �?              �?                      �?              �?              �?      �?              �?                      �?              �?      �?                      �?              �?              �?      �?              �?              �?                      �?              �?              �?      �?              �?                      �?              �?              �?              �?      �?                      �?      �?                      �?              �?      �?              �?                      �?      �?                      �?              �?              �?              �?      �?                      �?      �?                      �?              �?      �?                      �?      �?              �?                      �?              �?              �?      �?                      �?      �?              �?              �?                      �?      �?                      �?      �?              �?                      �?              �?              �?              �?      �?              �?              �?              �?              �?                      �?              �?              �?              �?              �?      �?                      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?                      �?              �?              �?              �?              �?      �?              �?                      �?              �?              �?              �?      �?                      �?              �?              �?      �?                      �?              �?      �?              �?                      �?      �?              �?              �?                      �?      �?              �?                      �?              �?      �?                      �?              �?              �?              �?              �?      �?                      �?              �?      �?              �?              �?                      �?              �?              �?              �?              �?              �?*
dtype0*
_output_shapes
:	�

�
softmax_cross_entropy_loss/CastCast!softmax_cross_entropy_loss/Cast/x*

SrcT0*
_output_shapes
:	�
*

DstT0
�
8softmax_cross_entropy_loss/xentropy/labels_stop_gradientStopGradientsoftmax_cross_entropy_loss/Cast*
T0*
_output_shapes
:	�

j
(softmax_cross_entropy_loss/xentropy/RankConst*
dtype0*
_output_shapes
: *
value	B :
z
)softmax_cross_entropy_loss/xentropy/ShapeConst*
dtype0*
_output_shapes
:*
valueB"u     
l
*softmax_cross_entropy_loss/xentropy/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
|
+softmax_cross_entropy_loss/xentropy/Shape_1Const*
valueB"u     *
dtype0*
_output_shapes
:
k
)softmax_cross_entropy_loss/xentropy/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'softmax_cross_entropy_loss/xentropy/SubSub*softmax_cross_entropy_loss/xentropy/Rank_1)softmax_cross_entropy_loss/xentropy/Sub/y*
T0*
_output_shapes
: 
�
/softmax_cross_entropy_loss/xentropy/Slice/beginPack'softmax_cross_entropy_loss/xentropy/Sub*
T0*

axis *
N*
_output_shapes
:
x
.softmax_cross_entropy_loss/xentropy/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
�
)softmax_cross_entropy_loss/xentropy/SliceSlice+softmax_cross_entropy_loss/xentropy/Shape_1/softmax_cross_entropy_loss/xentropy/Slice/begin.softmax_cross_entropy_loss/xentropy/Slice/size*
T0*
Index0*
_output_shapes
:
�
3softmax_cross_entropy_loss/xentropy/concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
q
/softmax_cross_entropy_loss/xentropy/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
*softmax_cross_entropy_loss/xentropy/concatConcatV23softmax_cross_entropy_loss/xentropy/concat/values_0)softmax_cross_entropy_loss/xentropy/Slice/softmax_cross_entropy_loss/xentropy/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
+softmax_cross_entropy_loss/xentropy/ReshapeReshapeadd_3*softmax_cross_entropy_loss/xentropy/concat*
_output_shapes
:	�
*
T0*
Tshape0
l
*softmax_cross_entropy_loss/xentropy/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
|
+softmax_cross_entropy_loss/xentropy/Shape_2Const*
dtype0*
_output_shapes
:*
valueB"u     
m
+softmax_cross_entropy_loss/xentropy/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
�
)softmax_cross_entropy_loss/xentropy/Sub_1Sub*softmax_cross_entropy_loss/xentropy/Rank_2+softmax_cross_entropy_loss/xentropy/Sub_1/y*
T0*
_output_shapes
: 
�
1softmax_cross_entropy_loss/xentropy/Slice_1/beginPack)softmax_cross_entropy_loss/xentropy/Sub_1*
T0*

axis *
N*
_output_shapes
:
z
0softmax_cross_entropy_loss/xentropy/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
�
+softmax_cross_entropy_loss/xentropy/Slice_1Slice+softmax_cross_entropy_loss/xentropy/Shape_21softmax_cross_entropy_loss/xentropy/Slice_1/begin0softmax_cross_entropy_loss/xentropy/Slice_1/size*
T0*
Index0*
_output_shapes
:
�
5softmax_cross_entropy_loss/xentropy/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
s
1softmax_cross_entropy_loss/xentropy/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
,softmax_cross_entropy_loss/xentropy/concat_1ConcatV25softmax_cross_entropy_loss/xentropy/concat_1/values_0+softmax_cross_entropy_loss/xentropy/Slice_11softmax_cross_entropy_loss/xentropy/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
-softmax_cross_entropy_loss/xentropy/Reshape_1Reshape8softmax_cross_entropy_loss/xentropy/labels_stop_gradient,softmax_cross_entropy_loss/xentropy/concat_1*
T0*
Tshape0*
_output_shapes
:	�

�
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits+softmax_cross_entropy_loss/xentropy/Reshape-softmax_cross_entropy_loss/xentropy/Reshape_1*
T0*&
_output_shapes
:�
:	�

m
+softmax_cross_entropy_loss/xentropy/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
�
)softmax_cross_entropy_loss/xentropy/Sub_2Sub(softmax_cross_entropy_loss/xentropy/Rank+softmax_cross_entropy_loss/xentropy/Sub_2/y*
T0*
_output_shapes
: 
{
1softmax_cross_entropy_loss/xentropy/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
�
0softmax_cross_entropy_loss/xentropy/Slice_2/sizePack)softmax_cross_entropy_loss/xentropy/Sub_2*
T0*

axis *
N*
_output_shapes
:
�
+softmax_cross_entropy_loss/xentropy/Slice_2Slice)softmax_cross_entropy_loss/xentropy/Shape1softmax_cross_entropy_loss/xentropy/Slice_2/begin0softmax_cross_entropy_loss/xentropy/Slice_2/size*
T0*
Index0*#
_output_shapes
:���������
�
-softmax_cross_entropy_loss/xentropy/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy+softmax_cross_entropy_loss/xentropy/Slice_2*
T0*
Tshape0*
_output_shapes	
:�

|
7softmax_cross_entropy_loss/assert_broadcastable/weightsConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
=softmax_cross_entropy_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
~
<softmax_cross_entropy_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
�
<softmax_cross_entropy_loss/assert_broadcastable/values/shapeConst*
dtype0*
_output_shapes
:*
valueB:�

}
;softmax_cross_entropy_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
S
Ksoftmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successNoOp
�
&softmax_cross_entropy_loss/ToFloat_1/xConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
softmax_cross_entropy_loss/MulMul-softmax_cross_entropy_loss/xentropy/Reshape_2&softmax_cross_entropy_loss/ToFloat_1/x*
T0*
_output_shapes	
:�

�
 softmax_cross_entropy_loss/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
:*
valueB: 
�
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
.softmax_cross_entropy_loss/num_present/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
,softmax_cross_entropy_loss/num_present/EqualEqual&softmax_cross_entropy_loss/ToFloat_1/x.softmax_cross_entropy_loss/num_present/Equal/y*
T0*
_output_shapes
: 
�
Asoftmax_cross_entropy_loss/num_present/zeros_like/shape_as_tensorConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
7softmax_cross_entropy_loss/num_present/zeros_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1softmax_cross_entropy_loss/num_present/zeros_likeFillAsoftmax_cross_entropy_loss/num_present/zeros_like/shape_as_tensor7softmax_cross_entropy_loss/num_present/zeros_like/Const*
_output_shapes
: *
T0*

index_type0
�
6softmax_cross_entropy_loss/num_present/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
6softmax_cross_entropy_loss/num_present/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
0softmax_cross_entropy_loss/num_present/ones_likeFill6softmax_cross_entropy_loss/num_present/ones_like/Shape6softmax_cross_entropy_loss/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
�
-softmax_cross_entropy_loss/num_present/SelectSelect,softmax_cross_entropy_loss/num_present/Equal1softmax_cross_entropy_loss/num_present/zeros_like0softmax_cross_entropy_loss/num_present/ones_like*
_output_shapes
: *
T0
�
[softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
�
Zsoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB:�
*
dtype0*
_output_shapes
:
�
Ysoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rankConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
�
isoftmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB:�
*
dtype0*
_output_shapes
:
�
Hsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_successj^softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_likeFillHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/ShapeHsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const*
_output_shapes	
:�
*
T0*

index_type0
�
8softmax_cross_entropy_loss/num_present/broadcast_weightsMul-softmax_cross_entropy_loss/num_present/SelectBsoftmax_cross_entropy_loss/num_present/broadcast_weights/ones_like*
T0*
_output_shapes	
:�

�
,softmax_cross_entropy_loss/num_present/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
�
&softmax_cross_entropy_loss/num_presentSum8softmax_cross_entropy_loss/num_present/broadcast_weights,softmax_cross_entropy_loss/num_present/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/Const_1ConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
$softmax_cross_entropy_loss/Greater/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/GreaterGreater&softmax_cross_entropy_loss/num_present$softmax_cross_entropy_loss/Greater/y*
T0*
_output_shapes
: 
�
"softmax_cross_entropy_loss/Equal/yConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
 softmax_cross_entropy_loss/EqualEqual&softmax_cross_entropy_loss/num_present"softmax_cross_entropy_loss/Equal/y*
T0*
_output_shapes
: 
�
*softmax_cross_entropy_loss/ones_like/ShapeConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
*softmax_cross_entropy_loss/ones_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$softmax_cross_entropy_loss/ones_likeFill*softmax_cross_entropy_loss/ones_like/Shape*softmax_cross_entropy_loss/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
�
!softmax_cross_entropy_loss/SelectSelect softmax_cross_entropy_loss/Equal$softmax_cross_entropy_loss/ones_like&softmax_cross_entropy_loss/num_present*
T0*
_output_shapes
: 
�
softmax_cross_entropy_loss/divRealDiv softmax_cross_entropy_loss/Sum_1!softmax_cross_entropy_loss/Select*
T0*
_output_shapes
: 
�
5softmax_cross_entropy_loss/zeros_like/shape_as_tensorConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
�
+softmax_cross_entropy_loss/zeros_like/ConstConstL^softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
%softmax_cross_entropy_loss/zeros_likeFill5softmax_cross_entropy_loss/zeros_like/shape_as_tensor+softmax_cross_entropy_loss/zeros_like/Const*
T0*

index_type0*
_output_shapes
: 
�
 softmax_cross_entropy_loss/valueSelect"softmax_cross_entropy_loss/Greatersoftmax_cross_entropy_loss/div%softmax_cross_entropy_loss/zeros_like*
T0*
_output_shapes
: 
^
OptimizeLoss/tagsConst*
dtype0*
_output_shapes
: *
valueB BOptimizeLoss
s
OptimizeLossScalarSummaryOptimizeLoss/tags softmax_cross_entropy_loss/value*
T0*
_output_shapes
: 
�
,mean/total/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
�
"mean/total/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
�
mean/total/Initializer/zerosFill,mean/total/Initializer/zeros/shape_as_tensor"mean/total/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@mean/total*
_output_shapes
: 
�

mean/total
VariableV2*
shared_name *
_class
loc:@mean/total*
	container *
shape: *
dtype0*
_output_shapes
: 
�
mean/total/AssignAssign
mean/totalmean/total/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@mean/total
g
mean/total/readIdentity
mean/total*
_output_shapes
: *
T0*
_class
loc:@mean/total
�
,mean/count/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
"mean/count/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
mean/count/Initializer/zerosFill,mean/count/Initializer/zeros/shape_as_tensor"mean/count/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@mean/count*
_output_shapes
: 
�

mean/count
VariableV2*
_class
loc:@mean/count*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
mean/count/AssignAssign
mean/countmean/count/Initializer/zeros*
use_locking(*
T0*
_class
loc:@mean/count*
validate_shape(*
_output_shapes
: 
g
mean/count/readIdentity
mean/count*
T0*
_class
loc:@mean/count*
_output_shapes
: 
K
	mean/SizeConst*
dtype0*
_output_shapes
: *
value	B :
Q
mean/ToFloat_1Cast	mean/Size*

SrcT0*
_output_shapes
: *

DstT0
M

mean/ConstConst*
dtype0*
_output_shapes
: *
valueB 
{
mean/SumSum softmax_cross_entropy_loss/value
mean/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
mean/AssignAdd	AssignAdd
mean/totalmean/Sum*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@mean/total
�
mean/AssignAdd_1	AssignAdd
mean/countmean/ToFloat_1!^softmax_cross_entropy_loss/value*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@mean/count
Z
mean/truedivRealDivmean/total/readmean/count/read*
T0*
_output_shapes
: 
b
mean/zeros_like/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
Z
mean/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
mean/zeros_likeFillmean/zeros_like/shape_as_tensormean/zeros_like/Const*
_output_shapes
: *
T0*

index_type0
Z
mean/GreaterGreatermean/count/readmean/zeros_like*
T0*
_output_shapes
: 
b

mean/valueSelectmean/Greatermean/truedivmean/zeros_like*
_output_shapes
: *
T0
\
mean/truediv_1RealDivmean/AssignAddmean/AssignAdd_1*
T0*
_output_shapes
: 
d
!mean/zeros_like_1/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
\
mean/zeros_like_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
mean/zeros_like_1Fill!mean/zeros_like_1/shape_as_tensormean/zeros_like_1/Const*
T0*

index_type0*
_output_shapes
: 
_
mean/Greater_1Greatermean/AssignAdd_1mean/zeros_like_1*
T0*
_output_shapes
: 
l
mean/update_opSelectmean/Greater_1mean/truediv_1mean/zeros_like_1*
T0*
_output_shapes
: 
#

group_depsNoOp^mean/update_op
�
+eval_step/Initializer/zeros/shape_as_tensorConst*
valueB *
_class
loc:@eval_step*
dtype0*
_output_shapes
: 
�
!eval_step/Initializer/zeros/ConstConst*
value	B	 R *
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
�
eval_step/Initializer/zerosFill+eval_step/Initializer/zeros/shape_as_tensor!eval_step/Initializer/zeros/Const*
T0	*

index_type0*
_class
loc:@eval_step*
_output_shapes
: 
�
	eval_step
VariableV2*
_class
loc:@eval_step*
	container *
shape: *
dtype0	*
_output_shapes
: *
shared_name 
�
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@eval_step*
validate_shape(*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
T0	*
_class
loc:@eval_step*
_output_shapes
: 
Q
AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@eval_step
U
readIdentity	eval_step^group_deps
^AssignAdd*
T0	*
_output_shapes
: 
;
IdentityIdentityread*
_output_shapes
: *
T0	
�
initNoOp^global_step/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedVariable*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitialized
Variable_1*
_class
loc:@Variable_1*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitialized
Variable_2*
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitialized
Variable_3*
_class
loc:@Variable_3*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitialized
Variable_4*
dtype0*
_output_shapes
: *
_class
loc:@Variable_4
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitialized
Variable_5*
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized
Variable_6*
dtype0*
_output_shapes
: *
_class
loc:@Variable_6
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized
Variable_7*
_class
loc:@Variable_7*
dtype0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitialized
mean/total*
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitialized
mean/count*
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitialized	eval_step*
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
�
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_11"/device:CPU:0*
N*
_output_shapes
:*
T0
*

axis 
�
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
�
$report_uninitialized_variables/ConstConst"/device:CPU:0*�
value�B�Bglobal_stepBVariableB
Variable_1B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7B
mean/totalB
mean/countB	eval_step*
dtype0*
_output_shapes
:
�
1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
�
3report_uninitialized_variables/boolean_mask/Shape_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 
�
Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_2StridedSlice3report_uninitialized_variables/boolean_mask/Shape_2Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackCreport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: 
�
;report_uninitialized_variables/boolean_mask/concat/values_1Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/strided_slice_1;report_uninitialized_variables/boolean_mask/concat/values_1;report_uninitialized_variables/boolean_mask/strided_slice_27report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*

Tidx0*
T0*
N*
_output_shapes
:
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
_output_shapes
:*
T0*
Tshape0
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:���������*
T0

�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*#
_output_shapes
:���������*
squeeze_dims
*
T0	
�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze"/device:CPU:0*
Tindices0	*
Tparams0*
validate_indices(*#
_output_shapes
:���������
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedVariable*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitialized
Variable_1*
_class
loc:@Variable_1*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitialized
Variable_2*
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitialized
Variable_3*
_class
loc:@Variable_3*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitialized
Variable_4*
dtype0*
_output_shapes
: *
_class
loc:@Variable_4
�
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitialized
Variable_5*
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitialized
Variable_6*
dtype0*
_output_shapes
: *
_class
loc:@Variable_6
�
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitialized
Variable_7*
_class
loc:@Variable_7*
dtype0*
_output_shapes
: 
�
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_8"/device:CPU:0*
T0
*

axis *
N	*
_output_shapes
:	
�
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:	
�
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*~
valueuBs	Bglobal_stepBVariableB
Variable_1B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7*
dtype0*
_output_shapes
:	
�
3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
valueB:	*
dtype0*
_output_shapes
:
�
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:
�
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:	
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
�
5report_uninitialized_variables_1/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:	*
dtype0*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_2StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_2Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
T0*
Index0
�
=report_uninitialized_variables_1/boolean_mask/concat/values_1Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
N*
_output_shapes
:*
T0*

axis 
�
9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/strided_slice_1=report_uninitialized_variables_1/boolean_mask/concat/values_1=report_uninitialized_variables_1/boolean_mask/strided_slice_29report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*
T0*
N*
_output_shapes
:*

Tidx0
�
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:	
�
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
���������
�
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:	
�
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:���������
�
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze"/device:CPU:0*
Tindices0	*
Tparams0*
validate_indices(*#
_output_shapes
:���������
I
init_2NoOp^mean/total/Assign^mean/count/Assign^eval_step/Assign

init_all_tablesNoOp

init_3NoOp
8
group_deps_2NoOp^init_2^init_all_tables^init_3
Q
Merge/MergeSummaryMergeSummaryOptimizeLoss*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_27990c85253b44348e02170c5b3644c2/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst"/device:CPU:0*~
valueuBs	BVariableB
Variable_1B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7Bglobal_step*
dtype0*
_output_shapes
:	
�
save/SaveV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable
Variable_1
Variable_2
Variable_3
Variable_4
Variable_5
Variable_6
Variable_7global_step"/device:CPU:0*
dtypes
2		
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
�
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*~
valueuBs	BVariableB
Variable_1B
Variable_2B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7Bglobal_step*
dtype0*
_output_shapes
:	
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2		*8
_output_shapes&
$:::::::::
�
save/AssignAssignVariablesave/RestoreV2*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
:	
�
�
save/Assign_1Assign
Variable_1save/RestoreV2:1*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*
_output_shapes	
:�
�
save/Assign_2Assign
Variable_2save/RestoreV2:2*
T0*
_class
loc:@Variable_2*
validate_shape(*
_output_shapes
:	�d*
use_locking(
�
save/Assign_3Assign
Variable_3save/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*
_output_shapes
:d
�
save/Assign_4Assign
Variable_4save/RestoreV2:4*
T0*
_class
loc:@Variable_4*
validate_shape(*
_output_shapes

:d2*
use_locking(
�
save/Assign_5Assign
Variable_5save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*
_output_shapes
:2
�
save/Assign_6Assign
Variable_6save/RestoreV2:6*
T0*
_class
loc:@Variable_6*
validate_shape(*
_output_shapes

:2*
use_locking(
�
save/Assign_7Assign
Variable_7save/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*
_output_shapes
:
�
save/Assign_8Assignglobal_stepsave/RestoreV2:8*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8
-
save/restore_allNoOp^save/restore_shard"" 
global_step

global_step:0"&

summary_op

Merge/MergeSummary:0"
	summaries

OptimizeLoss:0"�
trainable_variables��
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
Variable_7:0Variable_7/AssignVariable_7/read:02	zeros_3:0"
init_op

group_deps_1"U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0"
	eval_step

eval_step:0"2
metric_variables

mean/total:0
mean/count:0"�
local_variables��
T
mean/total:0mean/total/Assignmean/total/read:02mean/total/Initializer/zeros:0
T
mean/count:0mean/count/Assignmean/count/read:02mean/count/Initializer/zeros:0
P
eval_step:0eval_step/Assigneval_step/read:02eval_step/Initializer/zeros:0"!
local_init_op

group_deps_2"�
	variables��
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
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
Variable_7:0Variable_7/AssignVariable_7/read:02	zeros_3:0"
ready_op


concat:0"J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"0
losses&
$
"softmax_cross_entropy_loss/value:0��p�       ��-	_`9�j��Ad*

loss�Z�?í�