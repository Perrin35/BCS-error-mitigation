OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4709[3];
rz(0.39261079) q[0];
sx q[0];
rz(5.6568215) q[0];
sx q[0];
rz(9.4856833) q[0];
rz(1.159496) q[1];
sx q[1];
rz(-0.48519704) q[1];
sx q[1];
rz(-0.086866441) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
rz(1.5780014) q[2];
sx q[2];
rz(-0.8618827) q[2];
sx q[2];
rz(2.6345839) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-3.0806873) q[0];
sx q[0];
rz(-2.5152288) q[0];
sx q[0];
rz(2.7489819) q[0];
rz(0.5070088) q[1];
sx q[1];
rz(-0.8618827) q[1];
sx q[1];
rz(0.086866441) q[2];
sx q[2];
rz(-0.48519704) q[2];
sx q[2];
rz(-1.159496) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4709[0];
measure q[1] -> c4709[1];
measure q[2] -> c4709[2];
