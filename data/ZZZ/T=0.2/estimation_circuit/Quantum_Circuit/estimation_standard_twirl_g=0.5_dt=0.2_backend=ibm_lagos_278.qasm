OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4778[3];
rz(0.33534977) q[0];
sx q[0];
rz(-2.5367394) q[0];
sx q[0];
rz(-0.23515157) q[0];
rz(0.24808391) q[1];
sx q[1];
rz(-0.28548865) q[1];
sx q[1];
rz(-1.9774224) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
x q[1];
rz(0.79822531) q[2];
sx q[2];
rz(-0.38951482) q[2];
sx q[2];
rz(-2.2912765) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.9064411) q[0];
sx q[0];
rz(-0.60485324) q[0];
sx q[0];
rz(2.8062429) q[0];
rz(-0.85031616) q[1];
sx q[1];
rz(-0.38951482) q[1];
sx q[1];
rz(1.9774224) q[2];
sx q[2];
rz(-0.28548865) q[2];
sx q[2];
rz(-0.24808391) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4778[0];
measure q[1] -> c4778[1];
measure q[2] -> c4778[2];
