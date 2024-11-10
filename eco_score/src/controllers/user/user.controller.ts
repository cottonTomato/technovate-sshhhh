import { ReqHandler } from '../../types';
import { IAddUserDto } from './user.dto';
import { db } from '../../db/db';
import { user } from '../../db/schema';
import { StatusCodes } from 'http-status-codes';

type AddUserHandler = ReqHandler<IAddUserDto>;

export const addUserHandler: AddUserHandler = async function (req, res) {
  const userId = req.auth.userId!;
  const { firstname, lastname } = req.body;

  await db
    .insert(user)
    .values({ id: userId, firstName: firstname, lastName: lastname });

  res.status(StatusCodes.CREATED).json({
    status: 'Success',
  });
};
