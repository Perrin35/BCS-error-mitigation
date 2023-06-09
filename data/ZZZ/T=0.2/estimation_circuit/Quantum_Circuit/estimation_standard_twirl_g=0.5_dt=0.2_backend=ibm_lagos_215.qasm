OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4715[3];
rz(-0.013478181) q[0];
sx q[0];
rz(-0.38642281) q[0];
sx q[0];
rz(-2.7567182) q[0];
rz(-0.092777193) q[1];
sx q[1];
rz(-0.50748176) q[1];
sx q[1];
rz(0.98920554) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
rz(-1.5698655) q[2];
sx q[2];
rz(-2.2058827) q[2];
sx q[2];
rz(-2.5550856) q[2];
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
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
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
rz(-2.7567182) q[0];
sx q[0];
rz(-2.7551698) q[0];
sx q[0];
rz(-3.1281145) q[0];
rz(0.58650702) q[1];
sx q[1];
rz(0.93570999) q[1];
sx q[1];
rz(-0.98920554) q[2];
sx q[2];
rz(-0.50748176) q[2];
sx q[2];
rz(0.092777193) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4715[0];
measure q[1] -> c4715[1];
measure q[2] -> c4715[2];
