OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4762[3];
rz(2.2764533) q[0];
sx q[0];
rz(-1.1002914) q[0];
sx q[0];
rz(1.5143567) q[0];
rz(0.68669735) q[1];
sx q[1];
rz(-2.2795661) q[1];
sx q[1];
rz(1.0755737) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
rz(-2.8747332) q[2];
sx q[2];
rz(-1.8967532) q[2];
sx q[2];
rz(-2.9684249) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-1.6272359) q[0];
sx q[0];
rz(-2.0413013) q[0];
sx q[0];
rz(0.86513936) q[0];
rz(0.17316778) q[1];
sx q[1];
rz(-1.2448395) q[1];
sx q[1];
rz(-2.066019) q[2];
sx q[2];
rz(-0.86202654) q[2];
sx q[2];
rz(2.4548953) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4762[0];
measure q[1] -> c4762[1];
measure q[2] -> c4762[2];
