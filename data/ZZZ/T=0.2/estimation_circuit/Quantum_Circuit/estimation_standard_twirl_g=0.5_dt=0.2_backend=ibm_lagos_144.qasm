OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4644[3];
rz(3.6511143) q[0];
sx q[0];
rz(3.4283418) q[0];
sx q[0];
rz(12.313831) q[0];
rz(-0.11400873) q[1];
sx q[1];
rz(-1.9691363) q[1];
sx q[1];
rz(2.8199711) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
rz(3.080014) q[2];
sx q[2];
rz(-2.247844) q[2];
sx q[2];
rz(1.4358357) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.8890533) q[0];
sx q[0];
rz(-0.28674916) q[0];
sx q[0];
rz(-0.50952164) q[0];
rz(-1.4358357) q[1];
sx q[1];
rz(-2.247844) q[1];
sx q[1];
rz(2.8199711) q[2];
sx q[2];
rz(-1.1724563) q[2];
sx q[2];
rz(-3.0275839) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4644[0];
measure q[1] -> c4644[1];
measure q[2] -> c4644[2];
