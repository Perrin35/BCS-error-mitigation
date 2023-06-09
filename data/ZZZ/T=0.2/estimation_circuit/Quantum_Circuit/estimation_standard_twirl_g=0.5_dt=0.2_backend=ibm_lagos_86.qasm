OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4586[3];
rz(1.3396346) q[0];
sx q[0];
rz(3.1636163) q[0];
sx q[0];
rz(15.463762) q[0];
rz(-0.16744857) q[1];
sx q[1];
rz(-0.25820065) q[1];
sx q[1];
rz(2.3600654) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
rz(-1.2652994) q[2];
sx q[2];
rz(-2.1755887) q[2];
sx q[2];
rz(0.55917805) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-2.8973911) q[0];
sx q[0];
rz(-3.119569) q[0];
sx q[0];
rz(-1.3396346) q[0];
rz(-0.55917805) q[1];
sx q[1];
rz(2.1755887) q[1];
sx q[1];
rz(0.78152727) q[2];
sx q[2];
rz(-0.25820065) q[2];
sx q[2];
rz(0.16744857) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4586[0];
measure q[1] -> c4586[1];
measure q[2] -> c4586[2];
