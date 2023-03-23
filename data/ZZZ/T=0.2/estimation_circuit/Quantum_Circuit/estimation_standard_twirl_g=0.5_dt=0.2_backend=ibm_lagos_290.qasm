OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4790[3];
rz(0.0046303352) q[0];
sx q[0];
rz(-3.0807527) q[0];
sx q[0];
rz(2.2312387) q[0];
rz(-0.69801798) q[1];
sx q[1];
rz(-2.184977) q[1];
sx q[1];
rz(2.5024292) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
rz(-0.97597845) q[2];
sx q[2];
rz(-2.7739858) q[2];
sx q[2];
rz(-1.7473149) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.91035393) q[0];
sx q[0];
rz(-0.060839996) q[0];
sx q[0];
rz(3.1369623) q[0];
rz(-1.7473149) q[1];
sx q[1];
rz(-0.36760688) q[1];
sx q[1];
rz(-2.5024292) q[2];
sx q[2];
rz(-2.184977) q[2];
sx q[2];
rz(0.69801798) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4790[0];
measure q[1] -> c4790[1];
measure q[2] -> c4790[2];