OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4758[3];
rz(-1.7087348) q[0];
sx q[0];
rz(-2.1109348) q[0];
sx q[0];
rz(-0.090943737) q[0];
rz(0.2544075) q[1];
sx q[1];
rz(-0.15487691) q[1];
sx q[1];
rz(-1.9253908) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
x q[1];
rz(1.9580714) q[2];
sx q[2];
rz(-2.9289867) q[2];
sx q[2];
rz(-1.8837217) q[2];
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
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
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
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.090943737) q[0];
sx q[0];
rz(-1.0306579) q[0];
sx q[0];
rz(-1.4328579) q[0];
rz(1.8837217) q[1];
sx q[1];
rz(-2.9289867) q[1];
sx q[1];
rz(1.2162019) q[2];
sx q[2];
rz(-2.9867157) q[2];
sx q[2];
rz(2.8871852) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4758[0];
measure q[1] -> c4758[1];
measure q[2] -> c4758[2];
