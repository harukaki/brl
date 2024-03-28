from typing import Dict, List, Tuple

import numpy as np
from bridge_env import Pair, Player, Table
from bridge_env.data_handler.abstract_classes import BoardLog
from bridge_env.data_handler.json_handler.parser import JsonParser
from bridge_env.score import calc_score, score_to_imp
from omegaconf import OmegaConf
from pydantic import BaseModel


AnalysisResultDict = Dict[str, Dict[Table, Tuple[BoardLog, int]]]

class AnalyzeConfig(BaseModel):
    """Configuration settings for analyzing Board_log.

    Attributes:
        table1_results_path     File path for the results of table 1 in the duplicate match
        table2_results_path     File path for the results of table 2 in the duplicate match
        tag                     Tag for the team name of interest within the duplicate match
    """
    table1_results_path: str
    table2_results_path: str
    tag: str

args = AnalyzeConfig(**OmegaConf.to_object(OmegaConf.from_cli()))


class DuplicateBridgeAnalyzer:

    # TODO: Fix to use file_path1 as List[Path]
    def __init__(self, config):
        self._parser = JsonParser()

        with open(config["table1_results_path"],mode='r') as f:
            self.data1: List[BoardLog] = self._parser.parse_board_logs(f)

        with open(config["table2_results_path"], mode='r') as f:
            self.data2: List[BoardLog] = self._parser.parse_board_logs(f)

        print('============================================================')
        print(f'tag = {config["tag"]}')
        self._check_data_consistency()

        self.analysis_results: AnalysisResultDict = dict()

    def _check_data_consistency(self):
        self.n_name = self.data1[0].players[Player.N]
        self.e_name = self.data1[0].players[Player.E]
        for data in self.data1:
            assert data.players[Player.N] == self.n_name
            assert data.players[Player.E] == self.e_name
            assert data.players[Player.S] == self.n_name
            assert data.players[Player.W] == self.e_name
        for data in self.data2:
            assert data.players[Player.N] == self.e_name
            assert data.players[Player.E] == self.n_name
            assert data.players[Player.S] == self.e_name
            assert data.players[Player.W] == self.n_name
        print(f'Table1 : NS "{self.n_name}", EW "{self.e_name}"')
        print(f'Table2 : NS "{self.e_name}", EW "{self.n_name}"')

    def analyze(self) -> np.ndarray:
        self.analyze_board_logs(self.data1, Table.TABLE1, self.analysis_results)
        self.analyze_board_logs(self.data2, Table.TABLE2, self.analysis_results)

        imps = list()
        declarer_nums = {Table.TABLE1: {Pair.NS: 0, Pair.EW: 0},
                         Table.TABLE2: {Pair.NS: 0, Pair.EW: 0}}
        contract_levels: Dict[Table, Dict[Pair, List[int]]] = {
            table: {Pair.NS: [0, 0, 0, 0, 0, 0, 0],
                    Pair.EW: [0, 0, 0, 0, 0, 0, 0]} for table in Table}
        passed_out_num: Dict[Table, int] = {table: 0 for table in Table}
        double_num: Dict[Table, Dict[Pair, int]] = {
            table: {p: 0 for p in Pair} for table in Table}
        redouble_num: Dict[Table, Dict[Pair, int]] = {
            table: {p: 0 for p in Pair} for table in Table}

        for board_id, result in self.analysis_results.items():
            if Table.TABLE1 not in result or Table.TABLE2 not in result:
                print(f'data is loss. board id = {board_id}')
                continue

            board_log1, score1 = result[Table.TABLE1]
            if board_log1.declarer is not None:
                if board_log1.declarer.pair is Pair.EW:
                    score1 = - score1
            board_log2, score2 = result[Table.TABLE2]
            if board_log2.declarer is not None:
                if board_log2.declarer.pair is Pair.NS:
                    score2 = - score2

            imp = score_to_imp(score1, score2)
            imps.append(imp)

            for table in Table:
                board_log, _ = result[table]
                if not board_log.contract.is_passed_out():
                    level = board_log.contract.level
                    declarer = board_log.declarer
                    assert declarer is not None
                    assert level is not None

                    declarer_nums[table][declarer.pair] += 1
                    contract_levels[table][declarer.pair][level - 1] += 1

                    if board_log.contract.xx:
                        redouble_num[table][declarer.pair] += 1
                    elif board_log.contract.x:
                        double_num[table][declarer.pair] += 1

                else:
                    passed_out_num[table] += 1

        np_imps = np.array(imps)
        std = np_imps.std() / np.sqrt(len(np_imps))
        mean = np_imps.mean()
        print(f'Data num = {len(imps)}')
        print(f'IMP ave = '
              f'{self.n_name}: {mean} ± {std}, '
              f'{self.e_name}: {-mean} ± {std}')
        print(f'Declarer num: {declarer_nums}')
        print(f'Contract levels: {contract_levels}')
        print(f'Double: {double_num}')
        print(f'Redouble: {redouble_num}')
        print(f'Passed out {passed_out_num}')

        print(sorted(np_imps))
        return np_imps

    @staticmethod
    def analyze_board_logs(board_logs: List[BoardLog],
                           table: Table,
                           analysis_results: AnalysisResultDict) -> None:
        # analyze bidding phases with dda

        # players on the same place are same on all data
        for data in board_logs:
            board_id = data.board_id
            declarer = data.declarer
            contract = data.contract
            dda = data.dda
            assert dda is not None

            if contract.is_passed_out():
                dda_score = 0
            else:
                assert declarer is not None
                assert contract.trump is not None
                dda_score = calc_score(contract,
                                       dda[declarer][contract.trump])
            if board_id not in analysis_results:
                analysis_results[board_id] = dict()
            assert table not in analysis_results[board_id]
            analysis_results[board_id][table] = (data, dda_score)


def main() -> None:
    imps_list: List[np.ndarray] = list()
    tag_list: List[str] = list()
    for bidding_log in config.bidding_logs:
        duplicate_bridge_analyzer = DuplicateBridgeAnalyzer(bidding_log.path1,
                                                            bidding_log.path2,
                                                            bidding_log.tag)
        imps_list.append(duplicate_bridge_analyzer.analyze())
        tag_list.append(bidding_log.tag)



if __name__ == '__main__':
    config = {
        "table1_results_path":args.table1_results_path,
        "table2_results_path":args.table2_results_path,
        "tag":args.tag,
    }
    duplicate_bridge_analyzer = DuplicateBridgeAnalyzer(config)
    imps = duplicate_bridge_analyzer.analyze()