OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4508[3];
rz(2.7818662) q[0];
sx q[0];
rz(-2.5482228) q[0];
sx q[0];
rz(1.2523071) q[0];
rz(-2.4929555) q[1];
sx q[1];
rz(-2.5468002) q[1];
sx q[1];
rz(2.7313613) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(2.0992139) q[2];
sx q[2];
rz(-1.2487435) q[2];
sx q[2];
rz(-0.68795961) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-1.2523071) q[0];
sx q[0];
rz(-2.5482228) q[0];
sx q[0];
rz(-2.7818662) q[0];
rz(0.68795961) q[1];
sx q[1];
rz(-1.2487435) q[1];
sx q[1];
rz(-0.41023138) q[2];
sx q[2];
rz(-0.59479243) q[2];
sx q[2];
rz(-0.6486372) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4508[0];
measure q[1] -> c4508[1];
measure q[2] -> c4508[2];