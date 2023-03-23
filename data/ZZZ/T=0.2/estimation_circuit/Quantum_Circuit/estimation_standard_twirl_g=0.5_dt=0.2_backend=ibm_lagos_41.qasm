OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4541[3];
rz(5.1276418) q[0];
sx q[0];
rz(5.4045882) q[0];
sx q[0];
rz(15.378058) q[0];
rz(2.2613743) q[1];
sx q[1];
rz(-2.0236358) q[1];
sx q[1];
rz(2.1504724) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
x q[1];
rz(2.9410936) q[2];
sx q[2];
rz(-2.1110657) q[2];
sx q[2];
rz(0.021261332) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
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
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.8116871) q[0];
sx q[0];
rz(-2.2629955) q[0];
sx q[0];
rz(-1.9860492) q[0];
rz(0.021261332) q[1];
sx q[1];
rz(1.0305269) q[1];
sx q[1];
rz(0.99112029) q[2];
sx q[2];
rz(-2.0236358) q[2];
sx q[2];
rz(-2.2613743) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4541[0];
measure q[1] -> c4541[1];
measure q[2] -> c4541[2];
